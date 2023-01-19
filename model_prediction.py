#!/usr/bin/env python3

# 'Replace1', 'DfLv4', 'DfLv1', 'EX4', 'EX2', 'EX1', 'Source1', 'EX6',
# 'Country', 'EX11', 'NCause6', 'EX5', 'Gender', 'EX7', 'NCause5', 'EX3',
# 'Status', 'H9', 'EX9', 'NCause1', 'Replace5', 'RF1', 'Replace11', 'EX8',
# 'Replace2'

from pickle import load

import pandas as pd
from xgboost import XGBClassifier


def format_json(json: dict[str, str]) -> dict[str, dict[str, str]]:
    user_details = {
        'name': json['name'],
        'email': json['email']
    }

    features = {
        'Country': json['In which country do you live in?'],
        'DfLv1': json['The whole process'],
        'DfLv4': json['Choosing materials / products'],
        'EX1': json['Kitchen'],
        'EX2': json['Bathroom'],
        'EX3': json['Sewage'],
        'EX4': json['Windows'],
        'EX5': json['Floor'],
        'EX6': json['Facade'],
        'EX7': json['Additional Insulation - Attic'],
        'EX8': json['Additional Insulation - the basement'],
        'EX9': json['Additional Insulation - exterior walls'],
        'EX11': json['Heating System'],
        'H9': json['Type of Heating System'],
        'Gender': json['Gender'],
        'NCause1': json['I live in a newly built house'],
        'NCause5': json['I am statisfied with the standard of the house'],
        'NCause6': json['I am statisfied with the indoor climate'],
        'Replace1': json['It is too old'],
        'Replace2': json['It is damaged'],
        'Replace5': json['Change the layout / size'],
        'Replace11': json['Reduce energy cost'],
        'RF1': json['How should you finance the renovation?'],
        'Source1': json['Relatives / friends / neighbors'],
        'Status': json['Are you married/partner'],
    }

    return {
        'features': features,
        'user': user_details
    }


def clean_obj(json_obj: dict[str, str]) -> dict[str, int]:
    converter = {
        'Change': 1,
        'Maintaince': 2,
        'Yes': 1,
        'No': 2,
        'Man': 1,
        'Female': 2,
        'Other': 3,
        'Married': 1,
        'Not Married': 2
    }

    result: dict[str, int] = {}

    for key, value in json_obj.items():
        # change, maintaince
        if value in converter.keys():
            result[key] = converter[value]
        else:
            result[key] = int(json_obj[key])

    return result


def convert_result(n: int) -> str:
    # result = {
    #     1: 'No renovation at all',
    #     2: 'The whole at once',
    #     3: 'The whole house in steps',
    #     4: 'Some parts of the house'
    # }

    result = {
        1: 'You should not do renovation',
        2: 'You should do renovation all at once',
        3: 'You should do renovation but in steps',
        4: 'You should only renovate some part of the house'
    }

    return result[n]


def save_result(user: dict[str, str], result: str) -> None:
    with open('results.csv', 'a+') as file:
        line = f'{user["name"]},{user["email"]},{result}\n'
        file.write(line)


def save_answers(user: dict[str, str], answer_dataframe) -> None:
    answer_dataframe['name'] = user['name']
    answer_dataframe['email'] = user['email']

    answer_dataframe.to_csv('answer.csv', mode='a', index=False, header=False)


def predict(json_object: dict[str, str]) -> str:
    model = load(open('./asset/model.h5', 'rb'))

    formatted_object = format_json(json_object)
    cleaned_object = clean_obj(formatted_object['features'])
    data = pd.DataFrame(cleaned_object, index=[0])

    result = model.predict(data)
    save_result(formatted_object['user'], str(result[0]))
    save_answers(formatted_object['user'], data)

    return convert_result(result[0])
