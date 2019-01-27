import sys
sys.path.insert(0, '/home/ryo/cvpr2019/SQVolumetrics/')
from torch.autograd import Variable
from modules.transformer import rigidTsdf, rigidPointsTransform
from modules.quatUtils import quat_conjugate
from torch.nn import functional as F
import pdb
import torch


def cuboid_tsdf(sample_points, shape):
  ## sample_points Batch_size x nP x 3 , shape Batch_size x 1 x 3,
  ## output Batch_size x nP x 3
  nP = sample_points.size(1)
  shape_rep = shape.repeat(1, nP, 1)
  tsdf = torch.abs(sample_points) - shape_rep
  tsdfSq = F.relu(tsdf).pow(2).sum(dim=2)
  return tsdfSq  ## Batch_size x nP x 1

def sq_tsdf(sample_points, shape):
  ## sample_points Batch_size x nP x 3 , shape Batch_size x 1 x 5,
  ## output Batch_size x nP x 1
  nP = sample_points.size(1)
  shape_rep = shape.repeat(1, nP, 1)
  x = sample_points[:, :, 0]
  y = sample_points[:, :, 1]
  z = sample_points[:, :, 2]
  e1_rep = shape_rep[:, :, 0]
  e2_rep = shape_rep[:, :, 1]
  a1_rep = shape_rep[:, :, 2]
  a2_rep = shape_rep[:, :, 3]
  a3_rep = shape_rep[:, :, 4]

  term_1 = torch.pow(torch.div(x, a1_rep), torch.div(e2_rep, 2.0))
  term_2 = torch.pow(torch.div(y, a2_rep), torch.div(e2_rep, 2.0))
  term_3 = torch.pow(torch.div(z, a3_rep), torch.div(e1_rep, 2.0))
  ellip_term = torch.pow(torch.add(term_1, term_2), torch.div(e2_rep, e1_rep))
  sq_term = 1.0 - torch.pow(ellip_term + term_3, torch.div(e1_rep, -2.0))
  weight_term = torch.sqrt(torch.pow(x, 2.0) + torch.pow(y, 2.0) + torch.pow(z, 2.0))
  loss = weight_term * sq_term
  tsdfSq = F.relu(loss).pow(2)
  tsdfSq = torch.unsqueeze(tsdfSq, 2)
  return tsdfSq  ## Batch_size x nP x 1  

def tsdf_transform(sample_points, part):
  ## sample_points Batch_size x nP x 2, # parts Batch_size x 1 x 12
  shape = part[:, :, 0:5]  # B x 1 x 5
  trans = part[:, :, 5:8]  # B  x 1 x 3
  quat = part[:, :, 8:12]  # B x 1 x 4

  p1 = rigidTsdf(sample_points, trans, quat)  # B x nP x 3
  tsdf = sq_tsdf(p1, shape)  # B x nP x 1
  return tsdf


def get_existence_weights(tsdf, part):
  e = part[:,:,12:13]
  e = e.expand(tsdf.size())
  e = (1-e)*10
  return e


def tsdf_pred(sampledPoints, predParts):  ## coverage loss
  # sampledPoints  B x nP x 3
  # predParts  B x nParts x 12
  nParts = predParts.size(1)
  predParts = torch.chunk(predParts, nParts, dim=1)
  tsdfParts = []
  existence_weights = []
  for i in range(nParts):
    tsdf = tsdf_transform(sampledPoints, predParts[i])  # B x nP x 1
    tsdfParts.append(tsdf)
    existence_weights.append(get_existence_weights(tsdf, predParts[i]))

  existence_all = torch.cat(existence_weights, dim=2)
  tsdf_all = torch.cat(tsdfParts, dim=2) + existence_all
  tsdf_final = -1 * F.max_pool1d(-1 * tsdf_all, kernel_size=nParts)  # B x nP
  return tsdf_final


def primtive_surface_samples(predPart, sq_sampler):
  # B x 1 x 10
  shape = predPart[:, :, 0:5]  # B  x 1 x 5
  samples = sq_sampler.sample_points_sq(shape)
  #print(shape[0,:,:].clone().data.cpu())
  return samples


def partComposition(predParts, sq_sampler):
  # B x nParts x 13
  nParts = predParts.size(1)
  all_sampled_points = []
  predParts = torch.chunk(predParts, nParts, 1)
  for i in range(nParts):
    sampled_points = primtive_surface_samples(predParts[i], sq_sampler)
    transformedSamples = transform_samples(sampled_points, predParts[i])  # B x nPs x 3
    all_sampled_points.append(transformedSamples)  # B x nPs x 3

  pointsOut = torch.cat(all_sampled_points, dim=1)  # b x nPs*nParts x 3
  return pointsOut


def transform_samples(samples, predParts):
  # B x nSamples x 3  , predParts B x 1 x 12
  trans = predParts[:, :, 5:8]  # B  x 1 x 3
  quat = predParts[:, :, 8:12]  # B x 1 x 4
  transformedSamples = rigidPointsTransform(samples, trans, quat)
  return transformedSamples


def normalize_weights(imp_weights):
  # B x nP x 1
  totWeights = (torch.sum(imp_weights, dim=1) + 1E-6).repeat(1, imp_weights.size(1), 1)
  norm_weights = imp_weights / totWeights
  return norm_weights


def chamfer_loss(predParts, dataloader, sq_sampler):
  sampled_points = partComposition(predParts, sq_sampler)
  #print(sampled_points[0,:,:].clone().data.cpu())
  tsdfLosses = dataloader.chamfer_forward(sampled_points)
  weighted_loss = tsdfLosses  # B x nP x 1
  return torch.mean(weighted_loss, 1), sampled_points



def test_tsdf_pred():
  import numpy as np
  #pdb.set_trace()
  predParts = Variable(torch.FloatTensor([0.2, 0.2, 0.2, 0.2, 0.2,
                                          -0.2, -0.2, -0.2,
                                          0.5, np.sqrt(0.25), np.sqrt(0.25), np.sqrt(0.25), 0.5, 0.5
                                          ]).view(1,1,14))
  predParts = predParts.float()
  samplePoints = Variable(torch.FloatTensor([-0.4, -0.4, -0.4]).view(1,1,3))

  loss = tsdf_pred(samplePoints, predParts)
  print(loss)

if __name__=="__main__":
  test_tsdf_pred()