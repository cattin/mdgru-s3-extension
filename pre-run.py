#!/bin/python3
#
__author__ = "Robin Sandkuehler & Philippe Cattin"
__copyright__ = "Copyright (C) 2022 Uni Basel"

import boto3
import sys
import os
import json
import pandas as pd
import random
from omegaconf import OmegaConf
from omegaconf import DictConfig


def write_Json_File(ids, df, filename, featurefiles, maskfiles):
    # This function outputs a json file
    #
    jsonfile = {}

    for id in ids:
        entries = df[df['ID'] == id]
        for index, row in entries.iterrows():
            key = row['ID'] + '-' + row['AcqDate']
            if key not in jsonfile:
                jsonfile[key] = {"features": [], "labels": []}

            for feature in featurefiles:
                if feature in row['file']:
                    jsonfile[key]["features"].append(row['file'])
                    break

            for mask in maskfiles:
                if mask in row['file']:
                    jsonfile[key]["labels"].append(row['file'])
                    break

    with open(filename, "w") as f:
        json.dump(jsonfile, f, indent=4)


def check_Split(cfg):
    # Check the split
    train, val, test = cfg.datasplit.split
    if train+val+test != 100:
        print("The splits do not sum up to 100%")
        os._exit(1)

def read_S3_Storage(cfg, featurefiles, maskfiles):
    """
    This function reads the S3 bucket into a panda dataframe for easier processing
    """
    # Open connection to the S3 storage and provide the credentials
    s3 = boto3.client('s3', endpoint_url = cfg.lakefs.s3_endpoint,
                            aws_access_key_id = cfg.lakefs.access_key,
                            aws_secret_access_key = cfg.lakefs.secret_key)

    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket = cfg.lakefs.data_repository, Prefix = cfg.lakefs.branch)

    df = pd.DataFrame(columns=['ID', 'AcqDate', 'file'])

    n = 0
    for page in pages:
        for obj in page['Contents']:
            key = obj['Key']
            print(f"{n}", end='\r')
            n += 1

            # Ensure that the dataset (key) is really one in the list of featurefiles/maskfiles
            #   and possibly output a list of incomplete or wrong datasets
            flag = False
            for e in maskfiles + featurefiles:
                if e in key:
                    flag = True
                    break
            # If True add it to the DataFrame
            if flag:
                path,   file = os.path.split(key)
                path,   Acq  = os.path.split(path)
                branch, id   = os.path.split(path)
#                path,    file = os.path.split(key)
#                path, folder  = os.path.split(path)
                # get the patient ID and the Acq date from the folder name
#                id = folder.split("-")[1]
#                Acq = folder.split("-")[2]
                # Append the dataset to the dataframe
                newdata = pd.DataFrame(data = {'ID':[id] , 'AcqDate':[Acq] , 'file':[key]})
                df = pd.concat([df, newdata], axis=0)
            else:
                print("File '%s' does not seem to be one of the feature- or maskfiles." % key)
    return(df)


def run(cfg: DictConfig) -> None:
    """
    Reads the S3 storage bucket, splits the sets and saves them as JSON files
    """

    # extract the prefix from the filename
    featurefiles = []
    for feature in cfg.mdgru.featurefiles:
        featurefiles.append(feature.split(".")[0])
    maskfiles = []
    for mask in cfg.mdgru.maskfiles:
        maskfiles.append(mask.split(".")[0])

    # Reads the S3 content of the buckets into a dataframe
    df = read_S3_Storage(cfg, featurefiles, maskfiles)

    # Check if all scan dates indeed have all the required datasets
    # TODO: Make sure that all have all required datasets
    #  num_files_per_candidate_date =df[{'ID','AcqDate'}].value_counts()

    # Determine how many individual datasets are available for each patient ID
    #  (there should be a multiple of 3 i.e. the number of datasets per scan)
    counts = df['ID'].value_counts() # This is not needed but cool to know ;-)

    # Get a list of unique patient IDs
    ids = df['ID'].unique()
    print(ids)

    # Create the folder for the folds
    try:
        os.makedirs(cfg.datasplit.foldspath)
    except:
        pass

    for fold in range(cfg.datasplit.num_splits):
        # Create the folders
        try:
            os.makedirs( '%s/fold%d' % (cfg.datasplit.foldspath, fold+1) )
            os.makedirs( '%s/fold%d/train' % (cfg.datasplit.foldspath, fold+1) )
            os.makedirs( '%s/fold%d/val' % (cfg.datasplit.foldspath, fold+1) )
            os.makedirs( '%s/fold%d/test' % (cfg.datasplit.foldspath, fold+1) )
        except:
            pass

        # Shuffle this list
        random.shuffle(ids)

        # Convert the split-% into number of datasets per split and round
        #   Ensure that we indeed then include all datasets by tuning the size of the testing dataset
        num_train = int(round(len(ids)*cfg.datasplit.split[0]/100.))
        num_val   = int(round(len(ids)*cfg.datasplit.split[1]/100.))

        # Split the IDs in the required sizes
        train = ids[0:num_train]
        val   = ids[num_train:num_train+num_val]
        test  = ids[num_train+num_val:]

        write_Json_File(train, df, '%s/fold%d/train/train.json' % (cfg.datasplit.foldspath, fold+1),
                        featurefiles, maskfiles )
        write_Json_File(val  , df, '%s/fold%d/val/val.json' % (cfg.datasplit.foldspath, fold+1),
                        featurefiles, maskfiles)
        write_Json_File(test , df, '%s/fold%d/test/test.json' % (cfg.datasplit.foldspath, fold+1),
                        featurefiles, maskfiles)

    # Also write out the dataframe itself for debugging
    df.to_csv('/tmp/df-%s.csv' %  cfg.lakefs.data_repository)


if __name__ == "__main__":
    cli_conf = OmegaConf.from_cli()
    cfg = OmegaConf.load(cli_conf.config)
    cfg.datasplit.foldspath = os.path.expanduser(cfg.datasplit.foldspath)
    cfg.mdgru.datapath = os.path.expanduser(cfg.mdgru.datapath)
    cfg.mdgru.cache_path = os.path.expanduser(cfg.mdgru.cache_path)
    cfg.run.cache_path = os.path.expanduser(cfg.run.cache_path)

    # Set the seed for the random generator (if available) otherwise it throws an error but continues
    try:
        random.seed(cfg.datasplit.seed)
    except:
        pass

    # Check if the split between training, validation and testing adds up to 100
    check_Split(cfg)
    # Do the stuff
    run(cfg)
