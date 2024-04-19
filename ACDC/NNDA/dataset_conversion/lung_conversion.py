#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from collections import OrderedDict
from nnformer.paths import nnFormer_raw_data
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from multiprocessing import Pool
import nibabel


def reorient(filename):
    img = nibabel.load(filename)
    img = nibabel.as_closest_canonical(img)
    nibabel.save(img, filename)


if __name__ == "__main__":
    # base = "/media/fabian/DeepLearningData/Pancreas-CT"
    base = "/mnt/zqk/BTCV/model_test/SwinUNetR/ACDC/nnFormer-main/lung"

    # reorient
    p = Pool(8)
    results = []

    for f in subfiles(join(base, "images"), suffix=".nii.gz"):
        results.append(p.map_async(reorient, (f, )))
    _ = [i.get() for i in results]

    for f in subfiles(join(base, "labels"), suffix=".nii.gz"):
        results.append(p.map_async(reorient, (f, )))
    _ = [i.get() for i in results]

    task_id = 5
    task_name = "Lung"

    foldername = "Task%03.0d_%s" % (task_id, task_name)

    out_base = join(nnFormer_raw_data, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)

    train_patient_names = []
    test_patient_names = []
    # cases = list(range(1, 63))
    cases = [1,3,4,5,6,9,10,14,15,16,18,20,22,23,25,26,27,28,29,31,33,34,36,37,38,41,42,43,44,45,46,47,48,49,51,53,54,55,57,58,59,61,62,64,65,66,69,70,71,73,74,75,78,79,80,81,83,84,86,92,93,95,96]
    print('cases_len',len(cases))
    folder_data = join(base, "images")
    folder_labels = join(base, "labels")
    for c in cases:
        casename = "lung%03.0d" % c
        shutil.copy(join(folder_data, "lung_%03.0d.nii.gz" % c), join(imagestr, casename + "_0000.nii.gz"))
        shutil.copy(join(folder_labels, "lung_%03.0d.nii.gz" % c), join(labelstr, casename + ".nii.gz"))
        train_patient_names.append(casename)

    json_dict = OrderedDict()
    json_dict['name'] = task_name
    json_dict['description'] = task_name
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "see website"
    json_dict['licence'] = "see website"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "Lung cancer",
    }
    json_dict['numTraining'] = len(train_patient_names)
    json_dict['numTest'] = len(test_patient_names)
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i.split("/")[-1], "label": "./labelsTr/%s.nii.gz" % i.split("/")[-1]} for i in
                             train_patient_names]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i.split("/")[-1] for i in test_patient_names]

    save_json(json_dict, os.path.join(out_base, "dataset.json"))
