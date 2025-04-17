import torch
from repr_control.envs.models.articulate_model_fh import *
import matplotlib.pyplot as plt


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
    """
    check collision between two rectangles, batch version

    Args:
        rect1: [batch_size, 4, 2]
        rect2: [batch_size, 4, 2]

    Returns: [batch_size, 1] which is the distance.

    """
    if len(rect2.shape) == 2:
        rect2.unsqueeze_(0)
        rect2.expand_as(rect1)
    dist1 = collision_checking_one_direction(rect1, rect2)
    dist2 = collision_checking_one_direction(rect2, rect1)
    return torch.where(dist1 > dist2, dist1, dist2)  # select the larger one, if still lower then collision

def collision_checking_tt(states, rect_obstacles):
    """
    check collision between tractor-trailer (given states) and obstacles.
    Args:
        states: [batch_size, 6]
        rect_obstacles: [batch_size, 16, 2]

    Returns: dist, [batch_size, 1].

    """
    rect_tractor, rect_trailer = get_rectangles_tt(states)
    assert len(rect_obstacles.shape) == 3
    dists = []
    for i in range(4):
        obstacles = rect_obstacles[:, 4 * i : 4 * i + 4]
        dist_tractor = collision_checking_two_directions(rect_tractor, obstacles)
        dist_trailer = collision_checking_two_directions(rect_trailer, obstacles)
        dists.append(dist_tractor)
        dists.append(dist_trailer)
    dists = torch.vstack(dists).T
    min_dist = torch.min(dists, dim=1)[0]
    return min_dist



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
        eW = torch.vstack([-torch.sin(theta), torch.cos(theta)]).T.unsqueeze(1)

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


# def test_get_rectangles_tt():
#     states = initial_distribution(256)
#     rect_tractor, rect_trailor = get_rectangles_tt(states)
#     print(rect_tractor.shape, rect_trailor.shape)


def plot_rect(rect):
    from matplotlib.patches import Rectangle
    if isinstance(rect, torch.Tensor):
        rect = rect.numpy()
    # rectangle = Rectangle((x, y), width, height, edgecolor='blue', facecolor='none', linewidth=2)
    plt.scatter(rect[:, 0], rect[:, 1])


def test_collision_checking():
    import matplotlib.pyplot as plt
    states = initial_distribution(1)
    states[:, :3] -= torch.tensor([5.0, 5.0, 0.1])
    rect_tractor, rect_trailor = get_rectangles_tt(states)
    rect_obstacle = torch.tensor([[-40, -40, ],
                                  [-40, -5, ],
                                  [-5, -5, ],
                                  [-5, -40]]).float()
    dist1 = collision_checking_two_directions(rect_trailor, rect_obstacle)
    dist2 = collision_checking_two_directions(rect_tractor, rect_obstacle)
    fig, ax = plt.subplots()
    print(states)
    plot_rect(rect_tractor[0])
    plot_rect(rect_trailor[0])
    plot_rect(rect_obstacle[0])
    plt.axis('equal')
    plt.show()
    print(dist1, dist2)

def collision_checking_real():
    from repr_control.envs.models.articulate_model_fh import dynamics, rewards, initial_distribution
    states = initial_distribution(1)

    tt_states = states[:, :6]
    rect_tractor, rect_trailor = get_rectangles_tt(tt_states)
    obstacles = states[:, -32:]
    obstacles = obstacles.reshape([1, -1, 2])
    obstacles = torch.split(obstacles, 4, dim=1)
    dists = []
    for obstacle in obstacles:
        dist1 = collision_checking_two_directions(rect_tractor, obstacle)
        dist2 = collision_checking_two_directions(rect_trailor, obstacle)
        dists.append(dist1)
        dists.append(dist2)
    print(torch.vstack(dists))


if __name__ == '__main__':
    collision_checking_real()
