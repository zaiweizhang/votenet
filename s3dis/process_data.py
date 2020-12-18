import numpy as np
import os
import sys
sys.path.append('../utils')
from pc_util import *
from multiprocessing import Pool

data_path = 'Path/to/Stanford3dDataset_v1.2_Aligned_Version'
out_path = 'Path/to/output_dir'

if not os.path.exists(out_path):
    os.makedirs(out_path)
    
area_names = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_5', 'Area_6']
sem_map = {'ceiling':0, 'floor':1, 'wall':2, 'column':3,'beam':4, 'window':5, 'door':6,
           'table':7, 'chair':8, 'bookcase':9, 'sofa':10, 'board':11, 'clutter':12, 'stair':13}

def process_one_room(name_pair):
    area_name, room_name = name_pair.split('-')
    if os.path.exists(os.path.join(out_path, area_name+'_'+room_name+'_all_noangle.npy')):
        print('exists')
        return
    if not os.path.exists(os.path.join(data_path, area_name, room_name, room_name+'.txt')):
        return
    room_pt = np.loadtxt(os.path.join(data_path, area_name, room_name, room_name+'.txt'))
    #print(room_pt.shape)
    inds = np.random.choice(room_pt.shape[0], 50000)
    pt = room_pt[inds]
    # write_ply(pt[:,0:3], 'hh.ply')
    room_center = np.zeros((pt.shape[0], 3))
    room_size = np.zeros((pt.shape[0], 3))
    room_angle = np.zeros((pt.shape[0],1))
    room_ins = np.zeros((pt.shape[0],1))
    room_sem = np.zeros((pt.shape[0],1))
    print('start', area_name, room_name)
    for i, ins_name in enumerate(os.listdir(os.path.join(data_path, area_name, room_name, 'Annotations'))):
        print('{}/{}'.format(i, len(os.listdir(os.path.join(data_path, area_name, room_name, 'Annotations')))), area_name, room_name, ins_name)
        if not os.path.exists(os.path.join(data_path, area_name, room_name, 'Annotations', ins_name)):
            print('not exists', ins_name)
            continue
        
        #print(ins_name)
        sem_name = ins_name.split('_')[0]
        sem_label =  sem_map[sem_name]
        ins_label = i
        ins_pt = np.loadtxt(os.path.join(data_path, area_name, room_name, 'Annotations', ins_name))[:,0:3]
        x_min, y_min, z_min = ins_pt.min(0)
        x_max, y_max, z_max = ins_pt.max(0)
        size = np.array([x_max-x_min, y_max-y_min, z_max-z_min])
        center = np.array([x_max/2+x_min/2, y_max/2+y_min/2, z_max/2+z_min/2])
        ins_inds = []
        for j in range(ins_pt.shape[0]):
            dis = np.sum(np.square(ins_pt[j]-pt[:,0:3]),axis=1)
            min_dis = dis.min()
            min_ind = np.argmin(dis)
            if min_dis<1e-3:
                ins_inds.append(min_ind)
                
        room_center[ins_inds] = center
        room_size[ins_inds] = size
        room_ins[ins_inds] = ins_label+1
        room_sem[ins_inds] = sem_label+1
        
    print('prepare to save', area_name, room_name)
    res = np.concatenate((room_center, room_size, room_angle, room_ins, room_sem), axis=1)
    np.save(os.path.join(out_path, area_name+'_'+room_name+'_pt.npy'), pt)
    np.save(os.path.join(out_path, area_name+'_'+room_name+'_all_noangle.npy'), res)
    print('save', area_name, room_name)
    '''
    write_ply_color(pt[:,0:3], room_sem[:,0].astype(np.int32), 'test/'+area_name+'_'+room_nam
e+'_sem.ply')
    write_ply_color(pt[:,0:3], room_ins[:,0], 'test/'+area_name+'_'+room_name+'_ins.ply')
    centers = np.unique(room_center, axis=0)
    #centers = centers[np.sum(np.square(centers))>0]
    new_pt = np.concatenate((pt[:,0:3], centers), axis=0)
    colors = np.zeros(new_pt.shape[0])
    colors[pt.shape[0]:]=1
    write_ply_color(new_pt, colors, 'test/'+area_name+'_'+room_name+'_center.ply')
    print('save')
    '''
if __name__=="__main__":
    name_pairs = []
    for area_name in area_names:
        for room_name in os.listdir(os.path.join(data_path, area_name)):
            name_pairs.append(area_name+'-'+room_name)
            #process_one_room(area_name+'-'+room_name)
    p = Pool(12)
    p.map(process_one_room, name_pairs)
