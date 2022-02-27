from datetime import datetime

cat_features = [
       'brand', 'model_code', 'model_label',
       'commercial_label',  'article_main_category', 'article_type',
       'article_detail',  'color_code', 'color_label',
       'inaccurate_gender', 'country_of_origin', 'country_of_manufacture',
       'embakment_harbor', 'accurate_gender', 'size',
       'incorrect_fedas_code', 'correct_fedas_code'
]
time_features = [
       'avalability_start_date', 'avalability_end_date', 'shipping_date'
]
num_features = [
       'length', 'width', 'height',
       'eco_participation', 'eco_furniture', 'multiple_of_order',
       'minimum_multiple_of_order', 'net_weight', 'raw_weight', 'volume',
]

dict_values = {'time': (time_features, datetime.strptime(str(19700101), '%Y%m%d')),
               'categorical': (cat_features, ''),
               'numerical': (num_features, 0)
               }

target_column = 'correct_fedas_code'

columns_to_keep = num_features+time_features+[target_column]
