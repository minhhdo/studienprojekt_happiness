import pandas as pd

country_mapping = {
    1: "Austria", 2: "Belgium", 3: "Bulgaria", 4: "Cyprus", 5: "Czech Republic",
    6: "Germany", 7: "Denmark", 8: "Estonia", 9: "Greece", 10: "Spain",
    11: "Finland", 12: "France", 13: "Hungary", 14: "Ireland", 15: "Italy",
    16: "Lithuania", 17: "Luxembourg", 18: "Latvia", 19: "Malta", 20: "Netherlands",
    21: "Poland", 22: "Portugal", 23: "Romania", 24: "Sweden", 25: "Slovenia",
    26: "Slovakia", 27: "UK", 28: "Turkey", 29: "Croatia", 30: "Macedonia (FYROM)",
    31: "Kosovo", 32: "Serbia", 33: "Montenegro", 34: "Iceland", 35: "Norway"
}




def strict_question_filtering(input_dataframe):
    """
    This function takes the EQLS data and applies a strict filter.
    All questions that were answered in another question are removed
    as well as those that were had a lot of missing rows.
    """

    # Drop columns that are only used for statistical/analyzing purposes
    filtered_df = input_dataframe.drop(
        columns=['EQLS Wave', 'ISO3166_Country URL', 'RowID for the UK Data service Public API',
                 'Root URI for a row (respondent) that displays all data values for a single row via the UK Data Service Public API',
                 'Unique respondent ID'])

    filtered_df['Country'] = filtered_df['Country'].map(country_mapping)
    # One-Hot-Encoding
    filtered_df = pd.get_dummies(filtered_df, columns=['Country'], drop_first=True)

    # Remove question Y11_Q67_4 Citizenship - Don't know
    # Remove question Y11_Q67_4 Citizenship - Refusal
    # Reason: Less than 0.2% do not know their citizenship or refuse to answer
    filtered_df.drop(columns=['Citizenship - Don\'t know', 'Citizenship - Refusal'], inplace=True)

    # Remove Q: Degree of Urbanisation
    # Reason: 25% did not answer this question and question before answers almost the exact same
    filtered_df.drop(columns=['Degree of urbanisation'], inplace=True)

    # Fill Q: Direct contact with children
    # Reason: All empty rows are filled with 6-NA
    filtered_df['Direct contact with children'] = filtered_df['Direct contact with children'].fillna(6)

    # Fill Q: Direct contact with parents
    # Reason: All empty rows are filled with 6-NA
    filtered_df['Direct contact with parents'] = filtered_df['Direct contact with parents'].fillna(6)

    # Fill Q: Direct contact with neighbours
    # Reason: All empty rows are filled with 5-Never
    filtered_df['Direct contact with neighbours'] = filtered_df['Direct contact with neighbours'].fillna(5)

    # Fill Q: Phone/internet contact with children
    # Reason: All empty rows are filled with 6-NA
    filtered_df['Phone/internet contact with children'] = filtered_df['Phone/internet contact with children'].fillna(6)

    # Fill Q: Phone/internet contact with parents
    # Reason: All empty rows are filled with 6-NA
    filtered_df['Phone/internet contact with parents'] = filtered_df['Phone/internet contact with parents'].fillna(6)

    # Fill Q: Phone/internet contact with other relatives
    # Reason: All empty rows are filled with 6-NA
    filtered_df['Phone/internet contact with other relatives'] = filtered_df[
        'Phone/internet contact with other relatives'].fillna(6)

    # Fill Q: Phone/internet contact with other relatives
    # Reason: All empty rows are filled with 6-NA
    filtered_df['Phone/internet contact with other relatives'] = filtered_df[
        'Phone/internet contact with other relatives'].fillna(6)

    # Fill Q: Phone/internet contact with neighbours
    # Reason: All empty rows are filled with 5-Never
    filtered_df['Phone/internet contact with neighbours'] = filtered_df[
        'Phone/internet contact with neighbours'].fillna(5)

    # Remvoving Q
    # Reason: The questions are very specific and have very specific answering options
    filtered_df.drop(columns=['A person to get support from to help around house ',
                              'A person to get advice from about a personal/family matter',
                              'A person to get support from when looking for a job',
                              'A person to get support from when feeling depressed',
                              'A person to get support from to raise emergency money'], inplace=True)

    # Removing Q: No. of rooms in accommodation
    # Reason: Not relevant
    filtered_df.drop(columns=['No. of rooms in accommodation'], inplace=True)

    # Removing question about accommodation
    # Reason: Too specific and question 'No. of problems with accommodation' exists which generalizes this problem
    accommodation_q_to_remove = [
        "Problems with accommodation - space",
        "Problems with accommodation - rot in windows etc.",
        "Problems with accommodation - damp or leaks",
        "Problems with accommodation - no toilet",
        "Problems with accommodation - no bath/shower",
        "Problems with accommodation - no outside space",
        "Likelihood of leaving accom within 6 months",
        "Neighbourhood problems - noise",
        "Neighbourhood problems - air quality",
        "Neighbourhood problems - quality of drinking water",
        "Neighbourhood problems - crime, violence or vandalism",
        "Neighbourhood problems - litter or rubbish",
        "Neighbourhood problems - traffic"
    ]
    filtered_df.drop(columns=accommodation_q_to_remove, inplace=True, errors="ignore")

    # Removing question about Difficulty to see a doctor
    # Reason: Too specific and a lot of participants did not answer
    doctor_q_to_remove = [
        "Difficult to see a doctor because of distance?",
        "Difficult to see a doctor because of delay in getting appointment?",
        "Difficult to see a doctor because of waiting time?",
        "Difficult to see a doctor because of cost?",
        "Difficult to see a doctor because of lack of time?"
    ]
    filtered_df.drop(columns=doctor_q_to_remove, inplace=True, errors="ignore")

    # Removing these questions
    # Reason: Too many missing values
    quality_q_to_remove = [
        "Quality of child care services?",
        "Quality of long term care services?",
        "Quality of social/municipal housing?",
        "Quality of state pension system?"
    ]
    filtered_df.drop(columns=quality_q_to_remove, inplace=True, errors="ignore")

    # Removing these questions
    # Reason: Too many missing values, redundant questions since the first question "I or someone else in household used child care in 12 months" is enough
    childcare_q_to_remove = [
        'Someone close outside household used child care in 12 months',
        'Nobody used child care in 12 months',
        'Child care used in 12 months - don\'t know',
        'Child care used in 12 months - refusal',
    ]
    filtered_df.drop(columns=childcare_q_to_remove, inplace=True, errors="ignore")

    # Removing these questions
    # Reason: Too many missing values, redundant questions since the first question "I or someone else in household used long term care in 12" months is enough
    longtermcare_q_to_remove = [
        'Someone close outside household used long term care in 12 months',
        'Nobody used long term care in 12 months',
        'Long term care used in 12 months - don\'t know',
        'Long term care used in 12 months - refusal'
    ]
    filtered_df.drop(columns=longtermcare_q_to_remove, inplace=True, errors="ignore")

    # Removing these questions
    # Reason: Too many missing values, too specific
    tension_q_to_remove = [
        'How much tension between Poor and Rich?',
        'How much tension between Management and Workers?',
        'How much tension between Men and Women?',
        'How much tension between Old and Young people?',
        'How much tension between different racial/ethnic groups?',
        'How much tension between different religious groups?',
        'How much tension between groups with different sexual orientation?',
        'How much trust the parliament?',
        'How much trust the legal system?',
        'How much trust the press?',
        'How much trust the police?',
        'How much trust the local authorities?'
    ]
    filtered_df.drop(columns=tension_q_to_remove, inplace=True, errors='ignore')

    # Filling these questions
    # Reason: Not answering --> assume they do not go since >2% of the answers are missing
    frequency_q_to_fill = [
        'How frequently attend religious services?',
        'How frequently take part in sports or exercise?',
        'How frequently use the Internet other than for work?',
        'How frequently participate in social activities?'
    ]
    filtered_df[frequency_q_to_fill] = filtered_df[frequency_q_to_fill].fillna(5)

    # Removing these questions
    # Reason: Too many missing values, too specific
    activists_q_to_remove = [
        'How often worked unpaid for community services last 12 months?',
        'How often worked unpaid for education/cultural etc organisation last 12 months?',
        'How often worked unpaid for social movements/charities last 12 months?',
        'How often worked unpaid for political parties or trade unions last 12 months?',
        'How often worked unpaid for other voluntary org last 12 months?',
        'Attended a trade union/political party meeting last 12 months?',
        'Attended a protest or demonstration last 12 months?',
        'Signed a petition last 12 months?',
        'Contacted a politician last 12 months?'
    ]
    filtered_df.drop(columns=activists_q_to_remove, inplace=True, errors='ignore')

    # Removing these questions
    # Reason: Too many missing values, "Feel left out of sociey?", "Feel close to people in the area where I live" are the two poles for that topic, the rest is removed
    social_exclusion_q_to_remove = [
        'Can\'t find the way because life has become so complicated?',
        'The value of what I do is not recognised by others?',
        'People look down on me because of my job situation or income?',
        'Social Exclusion Index',
        'Volunteering frequency'
    ]
    filtered_df.drop(columns=social_exclusion_q_to_remove, inplace=True, errors='ignore')

    # Removing these questions
    # Reason: I would say these questions are included in 'Personal financial situation' and 'Household able to make ends meet?' --> The following are too specific
    household_bill_q_to_remove = [
        'Can afford to keep home adequately warm?',
        'Can afford to pay for a week\'s annual holiday away?',
        'Can afford to replace any worn-out furniture?',
        'Can afford a meal with meat/chicken/fish every second day?',
        'Can afford to buy new, rather than second-hand, clothes?',
        'Can afford to have friends or family for a drink/meal at least once a month?',
        'Rent/mortgage payments for accommodation',
        'Utility bills, such as electricity, water, gas',
        'Payments for consumer loans/credit cards',
        'Payments for informal loans from friends/relatives',
        'Financial situation of your household compared to 12 months ago?',
        'Household financial expectations for th 12 months?',
        'Deprivation index: No. of items hhold can\'t afford'
    ]
    filtered_df.drop(columns=household_bill_q_to_remove, inplace=True, errors='ignore')

    # Removing these questions
    # Reason: It is too close to the Happiness questions which I want to predict
    filtered_df.drop(columns='How satisfied with life these days?', inplace=True, errors='ignore')
    # removed_df.drop(columns='How happy are you?', inplace=True, errors='ignore')

    # Removing question beloging to the WHO mental well being index
    # Its using recent emotional states
    who_q_to_remove = [
        'How often felt cheerful and in good spirits last 2 weeks?',
        'How often felt calm and relaxed last 2 weeks?',
        'How often felt active and vigorous last 2 weeks?',
        'How often woke up feeling fresh and rested last 2 weeks?',
        'How often felt your daily life has been filled with things that interest you last 2 weeks?',
        'How often felt particularly tense last 2 weeks?',
        'How often felt lonely last 2 weeks?',
        'How often felt downhearted and depressed last 2 weeks?',
        'WHO-5 mental wellbeing index',
        'Final weight trimmed and standardised',
        'Cross-national weight - EU28 - to calculate averages for all EU in 2013 (incl. Croatia)',
        'Weight 5 total',
    ]
    filtered_df.drop(columns=who_q_to_remove, inplace=True, errors='ignore')

    # Remove these questions
    # Reason: Too many missing values for first question, second is removed due to consistency
    work_life_q_to_remove = [
        'How many hours per week would you prefer to work at present?',
        'How many hours per week would you prefer your partner to work?',
        'How often care for elderly or disabled relatives?',
        'The share of housework you do is?',
        # Missing values and 'Work-life balance conflict?' is a good question that summarizes these problems
        'As much time as would like with family members?',
        'As much time as would like with others (not family)?',
        'As much time as would like on own hobbies/interests?',
        'As much time as would like on voluntary work?'
    ]
    filtered_df.drop(columns=work_life_q_to_remove, inplace=True, errors='ignore')

    # Removing these questions
    # Reason: Duplicate questions, too close to other questions from before
    dv_q_to_remove = [
        'DV: Citizenship',
        'DV: Any limiting/not limiting chronic health problem?',
        'DV: Anyone used/would have like to use child care last 12 months?',
        'DV: Anyone used/would have like to use long term care last 12 months?',
        'DV: No. of factors which made it difficult to use child care?',
        'DV: No. of factors which made it difficult to use long term care?',
        'DV: Preferred working hours (3 groups)',
        'DV: Preferred working hours of respondent\'s partner? (3 groups)'
    ]
    filtered_df.drop(columns=dv_q_to_remove, inplace=True, errors='ignore')

    print(filtered_df.shape)
    return filtered_df


def loose_question_filtering(input_dataframe):
    """
    This function takes the EQLS data and applies a loose filter.
    Only questions that were used for statistical analysis were removed
    as well as questions that were missing a majority of the data.
    """

    # Question Removal
    # Drop columns that are only used for statistical/analyzing purposes
    filtered_df = input_dataframe.drop(
        columns=['EQLS Wave', 'ISO3166_Country URL', 'RowID for the UK Data service Public API',
                 'Root URI for a row (respondent) that displays all data values for a single row via the UK Data Service Public API',
                 'Unique respondent ID'])

    filtered_df['Country'] = filtered_df['Country'].map(country_mapping)
    # One-Hot-Encoding
    filtered_df = pd.get_dummies(filtered_df, columns=['Country'], drop_first=True)

    # Remove question Y11_Q67_4 Citizenship - Don't know
    # Remove question Y11_Q67_4 Citizenship - Refusal
    # Reason: Less than 0.2% do not know their citizenship or refuse to answer
    filtered_df.drop(columns=['Citizenship - Don\'t know', 'Citizenship - Refusal'], inplace=True)

    # Remove Q: Degree of Urbanisation
    # Reason: 25% did not answer this question and question before answers almost the exact same
    filtered_df.drop(columns=['Degree of urbanisation'], inplace=True)

    # Fill Q: Direct contact with children
    # Reason: All empty rows are filled with 6-NA
    filtered_df['Direct contact with children'] = filtered_df['Direct contact with children'].fillna(6)

    # Fill Q: Direct contact with parents
    # Reason: All empty rows are filled with 6-NA
    filtered_df['Direct contact with parents'] = filtered_df['Direct contact with parents'].fillna(6)

    # Fill Q: Direct contact with neighbours
    # Reason: All empty rows are filled with 5-NA
    filtered_df['Direct contact with neighbours'] = filtered_df['Direct contact with neighbours'].fillna(5)

    # Fill Q: Phone/internet contact with children
    # Reason: All empty rows are filled with 6-NA
    filtered_df['Phone/internet contact with children'] = filtered_df['Phone/internet contact with children'].fillna(6)

    # Fill Q: Phone/internet contact with parents
    # Reason: All empty rows are filled with 6-NA
    filtered_df['Phone/internet contact with parents'] = filtered_df['Phone/internet contact with parents'].fillna(6)

    # Fill Q: Phone/internet contact with other relatives
    # Reason: All empty rows are filled with 6-NA
    filtered_df['Phone/internet contact with other relatives'] = filtered_df[
        'Phone/internet contact with other relatives'].fillna(6)

    # Fill Q: Phone/internet contact with other relatives
    # Reason: All empty rows are filled with 6-NA
    filtered_df['Phone/internet contact with other relatives'] = filtered_df[
        'Phone/internet contact with other relatives'].fillna(6)

    # Fill Q: Phone/internet contact with neighbours
    # Reason: All empty rows are filled with 5-NA
    filtered_df['Phone/internet contact with neighbours'] = filtered_df[
        'Phone/internet contact with neighbours'].fillna(5)

    # Removing these questions
    # Reason: Too many missing values, too specific
    activists_q_to_remove = [
        'How often worked unpaid for community services last 12 months?',
        'How often worked unpaid for education/cultural etc organisation last 12 months?',
        'How often worked unpaid for social movements/charities last 12 months?',
        'How often worked unpaid for political parties or trade unions last 12 months?',
        'How often worked unpaid for other voluntary org last 12 months?',
        'Attended a trade union/political party meeting last 12 months?',
        'Attended a protest or demonstration last 12 months?',
        'Signed a petition last 12 months?',
        'Contacted a politician last 12 months?'
    ]
    filtered_df.drop(columns=activists_q_to_remove, inplace=True, errors='ignore')

    # Removing these questions
    # Reason: Too many missing values, "Feel left out of sociey?",
    # "Feel close to people in the area where I live" are the two poles for that topic, the rest is removed
    social_exclusion_q_to_remove = [
        # 'Can\'t find the way because life has become so complicated?',
        # 'The value of what I do is not recognised by others?',
        # 'People look down on me because of my job situation or income?',
        # 'Volunteering frequency',
        'Social Exclusion Index'
    ]
    filtered_df.drop(columns=social_exclusion_q_to_remove, inplace=True, errors='ignore')

    # Removing these questions
    # Reason: It is too close to the Happiness questions which I want to predict
    filtered_df.drop(columns='How satisfied with life these days?', inplace=True, errors='ignore')

    # Removing question beloging to the WHO mental well-being index
    # Its using recent emotional states
    who_q_to_remove = [
        'How often felt cheerful and in good spirits last 2 weeks?',
        'How often felt calm and relaxed last 2 weeks?',
        'How often felt active and vigorous last 2 weeks?',
        'How often woke up feeling fresh and rested last 2 weeks?',
        'How often felt your daily life has been filled with things that interest you last 2 weeks?',
        'How often felt particularly tense last 2 weeks?',
        'How often felt lonely last 2 weeks?',
        'How often felt downhearted and depressed last 2 weeks?',
        'WHO-5 mental wellbeing index',
        'Final weight trimmed and standardised',
        'Cross-national weight - EU28 - to calculate averages for all EU in 2013 (incl. Croatia)',
        'Weight 5 total',
    ]
    filtered_df.drop(columns=who_q_to_remove, inplace=True, errors='ignore')

    # Removing these questions
    # Reason: Duplicate questions, too close to other questions from before
    dv_q_to_remove = [
        'DV: Citizenship',
        'DV: Any limiting/not limiting chronic health problem?',
        'DV: Anyone used/would have like to use child care last 12 months?',
        'DV: Anyone used/would have like to use long term care last 12 months?',
        'DV: No. of factors which made it difficult to use child care?',
        'DV: No. of factors which made it difficult to use long term care?',
        'DV: Preferred working hours (3 groups)',
        'DV: Preferred working hours of respondent\'s partner? (3 groups)'
    ]
    filtered_df.drop(columns=dv_q_to_remove, inplace=True, errors='ignore')

    print(filtered_df.shape)
    return filtered_df

columns_to_remove = [
     'How satisfied with present standard of living?',
     'How satisfied with family life?',
     'How satisfied with social life?',
     'How satisfied with health?',
     'How satisfied with economic situation in the country?',
     'How satisfied with accommodation?',
     'How satisfied with education?'
]

columns_to_remove_extended = [
     'How satisfied with present standard of living?',
     'How satisfied with family life?',
     'How satisfied with social life?',
     'How satisfied with health?',
     'How satisfied with economic situation in the country?',
     'How satisfied with accommodation?',
     'How satisfied with education?',
    'Can\'t find the way because life has become so complicated?',
    'I feel I am free to decide how to live my life',
    'I generally feel that what I do in life is worthwhile',
    'I am optimistic about the future',
    'The value of what I do is not recognised by others?',
]