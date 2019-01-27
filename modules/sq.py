import torch
import torch.nn as nn
from torch.autograd import Variable
import math

class SQSurface(nn.Module):
  def __init__(self, nSamples, normFactor='None'):
    self.nSamples = nSamples

  def sample(self,dims, coeff):
    dims_rep = dims.repeat(1,self.nSamples, 1)
    e1, e2, a1, a2, a3 = torch.chunk(dims_rep, chunks=5, dim=2)
    x = a1 * torch.pow(torch.abs(torch.cos(coeff[:, :, 0])).unsqueeze(2), e1) *  torch.pow(torch.abs(torch.cos(coeff[:, :, 1])).unsqueeze(2), e2) 
    y = a2 * torch.pow(torch.abs(torch.cos(coeff[:, :, 0])).unsqueeze(2), e1) *  torch.pow(torch.abs(torch.sin(coeff[:, :, 1])).unsqueeze(2), e2) 
    z = a3 * torch.pow(torch.abs(torch.sin(coeff[:, :, 0])).unsqueeze(2), e1)
    point = torch.cat([x, y, z], dim=2)
    #point = Variable(point)
    return point


  def sample_points_sq(self, primShapes):
    # primPred B x 1 x 5
    # output B x nSamples x 3, B x nSamples x 1
    bs = primShapes.size(0)
    ns = self.nSamples

    data_type = primShapes.data.type()

    coeff = torch.Tensor(bs, ns, 2).type(data_type).uniform_(-1, 1)
    coeff[:, :, 0] = coeff[:, :, 0] * math.pi / 2.0 #theta
    coeff[:, :, 1] = coeff[:, :, 1] * math.pi / 2.0 #phi

    coeff = Variable(coeff)
    samples = self.sample(primShapes, coeff)

    return samples



import pdb
def test_sq_surface():
  N = 1
  P = 1

  sqSampler = SQSurface(18)
  primShapes  = torch.Tensor(N, P, 5).fill_(0.5)

  samples = sqSampler.sample_points_sq(primShapes)
  #pdb.set_trace()


if __name__ == "__main__":
  test_sq_surface()
