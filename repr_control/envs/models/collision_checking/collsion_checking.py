import torch
from repr_control.envs.models.articulate_model_fh import *
def collision_checking_one_direction(rect1, rect2):
    # %order of rectangle vertices
    # B ------ C
    # |        |
    # |        |
    # A ------ D

    # rect: numpy.array, [batch, 4, 2]

    # dist < 0: collision
    from torch.nn.functional import normalize

    A1 = rect1[:, [0], :]
    A1rect2 = rect2 - A1  # A1rect2: [bs, 4, 2]
    A1B1 = rect1[:, [1], :] - rect1[:, [0], :]  # A1B1 : [bs, 1, 2]

    def get_dist(vec):
        '''
        dist: [bs]
        '''
        vecn = normalize(vec, p=2, dim=2)
        rect2_to_rect1 = torch.multiply(A1rect2, vecn.expand(-1, 4, -1)).sum(dim=2)  # [bs, 4]
        dist_neg = - torch.max(rect2_to_rect1, dim=1)[0]  # dist to negative
        dist_pos = torch.min(rect2_to_rect1, dim=1)[0] - torch.norm(vec, p=2,
                                                                    dim=2).squeeze()  # [bs,]  # dist to positive
        dist = torch.where(dist_neg > dist_pos, dist_neg, dist_pos)
        return dist

    A1D1 = rect1[:, [3], :] - rect1[:, [0], :]  # A1B1 : [bs, 1, 2]

    dist_A1B1 = get_dist(A1B1)
    dist_A1D1 = get_dist(A1D1)
    return torch.where(dist_A1B1 > dist_A1D1, dist_A1B1, dist_A1D1)


def collision_checking_two_directions(rect1, rect2):
    if len(rect2.shape) == 2:
        rect2.unsqueeze_(0)
        rect2.expand_as(rect1)
    dist1 = collision_checking_one_direction(rect1, rect2)
    dist2 = collision_checking_one_direction(rect2, rect1)
    return torch.where(dist1 > dist2, dist1, dist2)  # select the larger one, if still lower then collision


def get_rectangles_tt(states):
    rw2b_car = 0.6096
    rw2b_trailor = 2.0320
    pos = states[:, :2].unsqueeze(1)
    theta0 = states[:, 2]
    theta1 = states[:, 2] + states[:, 3]

    def get_rect(pos, rw2b, theta, l, w):
        # pos: [bs, 1, 2]
        # theta: [bs,]
        eL = torch.vstack([torch.cos(theta), torch.sin(theta)]).T.unsqueeze(1)
        eW = torch.vstack([torch.cos(theta), torch.sin(theta)]).T.unsqueeze(1)

        def innerprod_dim1(batchvec1, batchvec2):
            # return a [batchsize, 4, 2] rectangles of car
            return torch.sum(torch.mul(batchvec1, batchvec2), dim=1)

        A_pos = pos - rw2b * eL - w * eW
        l_vec = eL * l
        w_vec = eW * w
        rects = torch.concat([A_pos,
                              A_pos + w_vec,
                              A_pos + w_vec + l_vec,
                              A_pos + l_vec], dim=1)
        return rects

    rect_tractor = get_rect(pos, rw2b_car, theta0, L_TRACTOR, W)

    # trailor
    eL_theta1 = torch.vstack([torch.cos(theta1), torch.sin(theta1)]).T.unsqueeze(1)
    pos_trailer = pos - L_TRAILER * eL_theta1
    rect_trailer = get_rect(pos_trailer, rw2b_trailor, theta1, L_TRAILER, W)

    return rect_tractor, rect_trailer

def global_to_local(states, inputs):
    '''
    convert rectangles from global frame to local frame,
    used for collision checking
    inputs: [bs, 4, 2]
    rotmat: [bs, 2, 2]
    '''

    pos, theta = states[:, :2], states[:, 2]
    pos.unsqueeze_(1)

    rotmat = torch.vstack([torch.cos(theta), -torch.sin(theta),
                           torch.sin(theta), torch.cos(theta)]).T.reshape(-1, 2, 2)
    rotmat_T = rotmat.transpose(1, 2)

    local_coord_pos = torch.bmm(inputs - pos, rotmat_T)
    return local_coord_pos

def test_get_rectangles_tt():
    states = initial_distribution(256)
    rect_tractor, rect_trailor = get_rectangles_tt(states)
    print(rect_tractor.shape, rect_trailor.shape)


def test_collision_checking():
    states = initial_distribution(4)
    rect_tractor, rect_trailor = get_rectangles_tt(states)
    rect_obstacle = torch.tensor([[-40, -40, ],
                                  [-40, -5, ],
                                  [-5, -5, ],
                                  [-5, -40]]).float()
    dist = collision_checking_two_directions(rect_trailor, rect_obstacle)
    print(dist)






if __name__ == '__main__':
    test_collision_checking()