
'''
CUDA_VISIBLE_DEVICES=1 python cadAutoEncCuboids/primSelTsdfChamfer.py
'''
import pdb
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../')))
import torch
import torch.nn as nn
import modules.volumeEncoder as vE
import modules.netUtils as netUtils
import modules.primitives as primitives

from torch.autograd import Variable
from data.cadConfigsChamfer import SimpleCadData
from modules.losses import tsdf_pred, chamfer_loss
#from modules.cuboid import  CuboidSurface
from modules.sq import SQSurface
import pdb
from modules.plotUtils import  plot3, plot_parts, plot_cuboid
import modules.marching_cubes as mc
import modules.meshUtils as mUtils
from modules.meshUtils import  savePredParts
from tensorboardX import SummaryWriter
torch.manual_seed(0)
from modules.config_utils import get_args
import numpy as np

params = get_args()

params.modelsDataDir = os.path.join('/home/ryo/volumetricPrimitives/cachedir/shapenet/chamferData/', params.synset)
params.visDir = os.path.join('../cachedir/visualization/', params.name)
params.visMeshesDir = os.path.join('../cachedir/visualization/meshes/', params.name)
params.snapshotDir = os.path.join('../cachedir/snapshots/', params.name)

logger = SummaryWriter('../cachedir/logs/{}/'.format(params.name))

dataloader = SimpleCadData(params)
params.primTypes = ['Sq']
params.nz = 5
params.nPrimChoices = len(params.primTypes)
params.intrinsicReward = torch.Tensor(len(params.primTypes)).fill_(0)

if not os.path.exists(params.visDir):
  os.makedirs(params.visDir)

if not os.path.exists(params.visMeshesDir):
  os.makedirs(params.visMeshesDir)

if not os.path.exists(params.snapshotDir):
  os.makedirs(params.snapshotDir)

params.primTypesSurface = []
for p in range(len(params.primTypes)):
    params.primTypesSurface.append(params.primTypes[p])

part_probs = []

sq_sampler = SQSurface(params.nSamplesChamfer, normFactor='Surf')
#cuboid_sampler = CuboidSurface(params.nSamplesChamfer, normFactor='Surf')
criterion  = nn.L1Loss()

import torch.nn as nn

class Network(nn.Module):
  def __init__(self, params):
    super(Network, self).__init__()
    self.ve = vE.convEncoderSimple3d(3,4,1,params.useBn)
    outChannels = self.outChannels = self.ve.output_channels
    layers = []
    #number of types of primitive shaped parameters (shape, scale, rot, trans)
    for i in range(2):
      layers.append(nn.Conv3d(outChannels, outChannels,kernel_size=1))
      layers.append(nn.BatchNorm3d(outChannels))
      layers.append(nn.LeakyReLU(0.2,True))

    self.fc_layers = nn.Sequential(*layers)
    self.fc_layers.apply(netUtils.weightsInit)

    biasTerms = lambda x:0

    biasTerms.quat = torch.Tensor([1, 0, 0, 0])
    biasTerms.shape = torch.Tensor(2).fill_(1) / params.shapeLrDecay
    biasTerms.scale = torch.Tensor(3).fill_(1)
    biasTerms.prob = torch.Tensor(1).fill_(0)
    for p in range(len(params.primTypes)):
      if (params.primTypes[p] == 'Sq'):
        biasTerms.prob[p] = 2.5 / params.probLrDecay

    self.primitivesTable = primitives.Primitives(params, outChannels, biasTerms)
    # self.primitivesTable.apply(netUtils.weightsInit)

  def forward(self, x):

    encoding  = self.ve(x)
    features = self.fc_layers(encoding)
    primitives, stocastic_actions = self.primitivesTable(features)
    return primitives, stocastic_actions


def train(dataloader, netPred, reward_shaper, optimizer, iter):
  inputVol, tsdfGt, sampledPoints, loaded_cps = dataloader.forward()
  # pdb.set_trace()
  inputVol = Variable(inputVol.clone().cuda())
  tsdfGt = Variable(tsdfGt.cuda())
  sampledPoints = Variable(sampledPoints.cuda()) ## B x np x 3
  predParts, stocastic_actions = netPred.forward(inputVol) ## B x nPars*13
  predParts = predParts.view(predParts.size(0), params.nParts, 14)
  optimizer.zero_grad()
  tsdfPred = tsdf_pred(sampledPoints, predParts)
  # coverage = criterion(tsdfPred, tsdfGt)
  coverage_b = tsdfPred.mean(dim=1).squeeze()
  coverage = coverage_b.mean()
  consistency_b, sampled_points = chamfer_loss(predParts, dataloader, sq_sampler)
  #sampled_points = sampled_points.clone().data.cpu().numpy()[0, :, :]
  #np.savetxt('./log_' + str(iter) + '.txt',  sampled_points, delimiter=',')
  consistency_b = consistency_b.squeeze()
  consistency = consistency_b.mean()
  loss = coverage_b + params.chamferLossWt*consistency_b
  rewards = []
  mean_reward = 0
  #if params.prune:
  if params.prune == 1:
    reward = -1*loss.view(-1,1).data
    for i, action in enumerate(stocastic_actions):
      shaped_reward = reward - params.nullReward*torch.sum(action.data)
      shaped_reward = reward_shaper.forward(shaped_reward)
      action.reinforce(shaped_reward)
      rewards.append(shaped_reward)

    mean_reward = torch.stack(rewards).mean()

  logger.add_scalar('rewards/', mean_reward, iter)
  for i in range(params.nParts):
    logger.add_scalar('{}/prob'.format(i), predParts[:,i,-1].data.mean(), iter)
  logger.add_scalar('total_loss', loss.data[0], iter)
  logger.add_scalar('coverage loss', coverage.data[0], iter)
  logger.add_scalar('consistency loss', consistency.data[0])

  loss = torch.mean(loss)
  loss.backward()
  optimizer.step()

  return loss.data[0], coverage.data[0], consistency.data[0], mean_reward




netPred = Network(params)
netPred.cuda()

reward_shaper = primitives.ReinforceShapeReward(params.bMomentum, params.intrinsicReward, params.entropyWt)
optimizer = torch.optim.Adam(netPred.parameters(), lr=params.learningRate)

nSamplePointsTrain = params.nSamplePoints
nSamplePointsTest = params.gridSize**3

loss = 0
coverage = 0
consistency = 0
mean_reward = 0


def tsdfSqModTest(x):
  return torch.clamp(x,min=0).pow(2)



print("Iter\tErr\tTSDF\tChamf\tMeanRe")
for iter  in range(params.numTrainIter):
  print("{:10.7f}\t{:10.7f}\t{:10.7f}\t{:10.7f}\t{:10.7f}".format(iter, loss, coverage, consistency, mean_reward))
  loss, coverage, consistency, mean_reward = train(dataloader, netPred, reward_shaper, optimizer, iter)
  netPred.train()


  '''
  if iter % params.visIter ==0:
    reshapeSize = torch.Size([params.batchSizeVis, 1, params.gridSize, params.gridSize, params.gridSize])

    sample, tsdfGt, sampledPoints = dataloader.forwardTest()

    sampledPoints = sampledPoints[0:params.batchSizeVis].cuda()
    sample = sample[0:params.batchSizeVis].cuda()
    tsdfGt = tsdfGt[0:params.batchSizeVis].view(reshapeSize)

    tsdfGtSq = tsdfSqModTest(tsdfGt)
    netPred.eval()
    shapePredParams, _ = netPred.forward(Variable(sample))
    
    shapePredParams = shapePredParams.view(params.batchSizeVis, params.nParts, 14)

    netPred.train()

    if iter % params.meshSaveIter ==0:

      meshGridInit = primitives.meshGrid([-params.gridBound, -params.gridBound, -params.gridBound],
                                         [params.gridBound, params.gridBound, params.gridBound],
                                         [params.gridSize, params.gridSize, params.gridSize])
      predParams = shapePredParams
      for b in range(0, tsdfGt.size(0)):

        visTriSurf = mc.march(tsdfGt[b][0].cpu().numpy())
        mc.writeObj('{}/iter{}_inst{}_gt.obj'.format(params.visMeshesDir ,iter, b), visTriSurf)


        pred_b = []
        for px in range(params.nParts):
          pred_b.append(predParams[b,px,:].clone().data.cpu())

        mUtils.saveParts(pred_b, '{}/iter{}_inst{}_pred.obj'.format(params.visMeshesDir, iter, b))

  if ((iter+1) % 1000) == 0 :
    torch.save(netPred.state_dict() ,"{}/iter{}.pkl".format(params.snapshotDir,iter))
  '''
