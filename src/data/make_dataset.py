# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    df1 = pd.read_csv(
        "A:\\Practice_Project\\MachineLearningPractice\\CreditRiskModeling\\data\\interim\\case_study1_Internal.csv")
    df2 = pd.read_csv(
        "A:\\Practice_Project\\MachineLearningPractice\\CreditRiskModeling\\data\\external\\case_study2_External.csv")

    # working on dataframe 1
    df1 = df1.loc[df1['Age_Oldest_TL'] != -99999]

    # working on dataframe 2
    columnToBeRemoved = []
    for i in df2.columns:
        if df2.loc[df2[i] == -99999].shape[0] > 10000:
            columnToBeRemoved.append(i)

    df2.drop(columnToBeRemoved, axis=1, inplace=True)

    for i in df2.columns:
        df2.drop(df2.loc[df2[i] == -99999].index, inplace=True)

    # Merging 2 data sets
    df = pd.merge(df1, df2, how='inner', left_on=['PROSPECTID'], right_on=['PROSPECTID'])

    df.to_csv("A:\\Practice_Project\\MachineLearningPractice\\CreditRiskModeling\\data\\raw\\df_merged.csv",
              index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    # project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
