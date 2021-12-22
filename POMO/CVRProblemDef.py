
import torch
import numpy as np
import datetime

def get_random_problems(batch_size, problem_size):

    depot_xy = torch.rand(size=(batch_size, 1, 2))
    # shape: (batch, 1, 2)

    # node_xy = torch.rand(size=(batch_size, problem_size, 2))
    # # shape: (batch, problem, 2)
    # 시간
    node_x = torch.randint(1, 24, size=(batch_size, problem_size, 1)) #/ float(41024)
    # 분
    node_y = torch.randint(1, 60, size=(batch_size, problem_size, 1)) #/ float(60)    
    # 시간, 분
    node_xy = torch.cat((node_x, node_y), dim = 2)
    # unix time
    node_unixtime = torch.zeros(batch_size, problem_size, 3, dtype = torch.long) 
    # 0    1    2    3    4    5 
    # time time time time time time 

    # 초기 값 날짜는 임의로 설정
    # 현재는 시간과 분만 랜덤으로 발생
    date = '2021-01-01 00:00:00'

    # convert string to datetimeformat
    date = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    
    if problem_size == 20:
        demand_scaler = 30
    elif problem_size == 50:
        demand_scaler = 40
    elif problem_size == 100:
        demand_scaler = 50
    else:
        raise NotImplementedError

    # 노드별 작업 시간 (단위: 분)
    node_demand = torch.randint(40, 60, size=(batch_size, problem_size)) / float(demand_scaler)

    for i, x in enumerate(node_xy[0]):
        date = date.replace(hour = node_xy[0][i][0], minute = node_xy[0][i][1])
    
        node_unixtime[0][i][0] = i
        # 작업 시작 시간
        node_unixtime[0][i][1] = date.timestamp()
        # 작업 종료 시간
        node_unixtime[0][i][2] = node_unixtime[0][i][1] + (node_demand[i] * 60)
        print(i, node_xy[0][i][0], node_xy[0][i][1], date.strftime("%Y-%m-%d %H:%M:%S"), node_unixtime[0][i][1], node_unixtime[0][i][2])
            



    # shape: (batch, problem)

    node_xy_unixtime = torch.cat((node_xy, node_unixtime), dim = 2)
    print(node_xy_unixtime)
    return depot_xy, node_xy, node_demand, node_unixtime


def augment_xy_data_by_8_fold(xy_data):
    # xy_data.shape: (batch, N, 2)

    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    # x,y shape: (batch, N, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, N, 2)

    return aug_xy_data