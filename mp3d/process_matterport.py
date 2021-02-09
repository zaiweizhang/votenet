import numpy as np
import os
from plyfile import PlyData, PlyElement
import json
import sys
import csv
from matplotlib import cm

def read_mesh_vertices_rgb(filename):
    """ read XYZ RGB for each vertex.
    Note: RGB values are in 0-255
    """
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
        vertices[:,3] = plydata['vertex'].data['red']
        vertices[:,4] = plydata['vertex'].data['green']
        vertices[:,5] = plydata['vertex'].data['blue']
    return vertices

def write_mesh_vertices_rgb(filename, out_name, colors):
    """ read XYZ RGB for each vertex.
    Note: RGB values are in 0-255
    """
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count

        plydata['vertex'].data['red'] = colors[:,0]
        plydata['vertex'].data['green'] = colors[:,1]
        plydata['vertex'].data['blue'] = colors[:,2]
        plydata.write(out_name)

def read_aggregation(filename):
    assert os.path.isfile(filename)
    object_id_to_segs = {}
    label_to_segs = {}
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data['segGroups'])
        for i in range(num_objects):
            object_id = data['segGroups'][i]['objectId'] + 1 # instance ids should be 1-indexed
            label = data['segGroups'][i]['label']
            segs = data['segGroups'][i]['segments']
            object_id_to_segs[object_id] = segs
            if label in label_to_segs:
                label_to_segs[label].extend(segs)
            else:
                label_to_segs[label] = segs
    return object_id_to_segs, label_to_segs

def read_segmentation(filename):
    assert os.path.isfile(filename)
    seg_to_verts = {}
    with open(filename) as f:
        data = json.load(f)
        num_verts = len(data['segIndices'])
        for i in range(num_verts):
            seg_id = data['segIndices'][i]
            if seg_id in seg_to_verts:
                seg_to_verts[seg_id].append(i)
            else:
                seg_to_verts[seg_id] = [i]
    return seg_to_verts, num_verts

def represents_int(s):
    ''' if string s represents an int. '''
    try:
        int(s)
        return True
    except ValueError:
        return False
def read_label_mapping(filename, label_from='raw_category', label_to='nyu40id'):
    assert os.path.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            if row[label_from]=='' or row[label_to]=='':
                continue
            mapping[row[label_from]] = int(row[label_to])
    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k):v for k,v in mapping.items()}
    return mapping

data_path = 'scans'
out_path = 'ours'
if not os.path.exists(out_path):
    os.makedirs(out_path)

LABEL_MAP_FILE = '/home/bo/data/matterport3d/Matterport/metadata/category_mapping.tsv'
label_map = read_label_mapping(LABEL_MAP_FILE,
        label_from='raw_category', label_to='nyu40id')


for house_name in os.listdir(data_path):
    for tmp_name in os.listdir(os.path.join(data_path, house_name,  'region_segmentations')):
        if not tmp_name.endswith('.ply'):
            continue
        region_name = tmp_name[:-4]
        print(house_name, region_name)
        mesh_vertices = read_mesh_vertices_rgb(os.path.join(data_path,house_name, 'region_segmentations', region_name+'.ply'))
        fseg_filename = os.path.join(data_path, house_name, 'region_segmentations', region_name+'.fsegs.json')
        vseg_filename = os.path.join(data_path, house_name, 'region_segmentations', region_name+'.vsegs.json')
        semseg_filename = os.path.join(data_path, house_name, 'region_segmentations', region_name+'.semseg.json')
        with open(semseg_filename) as f:
            try:
                data = json.load(f)
            except:
                print('bad json file', semseg_filename)
                continue
        with open(vseg_filename) as f:
            try:
                data = json.load(f)
            except:
                print('bad json file', vseg_filename)
                continue
        object_id_to_segs, label_to_segs = read_aggregation(semseg_filename)
        seg_to_verts, num_verts = read_segmentation(vseg_filename)

        label_ids = np.zeros(shape=(num_verts), dtype=np.uint32) # 0: unannotated
        object_id_to_label_id = {}
        for label, segs in label_to_segs.items():

            label_id = label_map[label]
            for seg in segs:
                if not seg in seg_to_verts.keys():
                    continue
                verts = seg_to_verts[seg]
                label_ids[verts] = label_id
        instance_ids = np.zeros(shape=(num_verts), dtype=np.uint32) # 0: unannotated
        num_instances = len(np.unique(list(object_id_to_segs.keys())))
        for object_id, segs in object_id_to_segs.items():
            for seg in segs:
                if not seg in seg_to_verts.keys():
                     continue
                verts = seg_to_verts[seg]
                instance_ids[verts] = object_id
                if object_id not in object_id_to_label_id:
                    object_id_to_label_id[object_id] = label_ids[verts][0]
            instance_bboxes = np.zeros((num_instances,7))

        for obj_id in object_id_to_segs:
            label_id = object_id_to_label_id[obj_id]
            obj_pc = mesh_vertices[instance_ids==obj_id, 0:3]
            if len(obj_pc) == 0: continue
            xmin = np.min(obj_pc[:,0])
            ymin = np.min(obj_pc[:,1])
            zmin = np.min(obj_pc[:,2])
            xmax = np.max(obj_pc[:,0])
            ymax = np.max(obj_pc[:,1])
            zmax = np.max(obj_pc[:,2])
            bbox = np.array([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2,
                xmax-xmin, ymax-ymin, zmax-zmin, label_id])
            # NOTE: this assumes obj_id is in 1,2,3,.,,,.NUM_INSTANCES
            instance_bboxes[obj_id-1,:] = bbox

        output_file = '{}/{}_{}'.format(out_path, house_name, region_name)
        """
        c = label_ids.astype(np.float32)/label_ids.max()
        cmap=cm.get_cmap('rainbow')
        colors = cmap(c)[:,0:3]*255.0

        write_mesh_vertices_rgb(os.path.join(data_path,house_name, 'region_segmentations', region_name+'.ply'),
                output_file+'_sem.ply', colors)
        c = instance_ids.astype(np.float32)/instance_ids.max()
        cmap=cm.get_cmap('rainbow')
        colors = cmap(c)[:,0:3]*255.0
        write_mesh_vertices_rgb(os.path.join(data_path,house_name, 'region_segmentations', region_name+'.ply'),
                 output_file+'_ins.ply', colors)
        """
        np.save(output_file+'_vert.npy', mesh_vertices)
        np.save(output_file+'_sem_label.npy', label_ids)
        np.save(output_file+'_ins_label.npy', instance_ids)
        np.save(output_file+'_bbox.npy', instance_bboxes)


        # with open(fseg_filename) as f:
        #     data = json.load(f)
        #     fseg_ids = data['segIndices']

        # object_id_to_segs = {}
        # label_to_segs = {}
        # with open(semseg_filename) as f:
        #     data = json.losd(f)
        #     num_objects = len(data['segGroups'])
        #     for i in range(num_objects):
        #         object_id = data['segGroups'][i]['objectId'] + 1 # instance ids should be 1-indexed
        #         label = data['segGroups'][i]['label']
        #         segs = data['segGroups'][i]['segments']

        #         object_id_to_segs[object_id] = segs
        #         if label in label_to_segs:
        #             label_to_segs[label].extend(segs)
        #         else:
        #             label_to_segs[label] = segs



    # os.system('unzip %s'%(os.path.join(data_path, house_name, 'region_segmentations.zip')))
