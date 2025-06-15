#!/bin/bash

# This script parses all the .v files in the release directories and generates corresponding .csv files in the training_data directory.
python3 parser.py release\(20250520\)/release/design0.v training_data/design0.csv
python3 parser.py release\(20250520\)/release/design1.v training_data/design1.csv
python3 parser.py release\(20250520\)/release/design2.v training_data/design2.csv
python3 parser.py release\(20250520\)/release/design3.v training_data/design3.csv
python3 parser.py release\(20250520\)/release/design4.v training_data/design4.csv
python3 parser.py release\(20250520\)/release/design5.v training_data/design5.csv
python3 parser.py release\(20250520\)/release/design6.v training_data/design6.csv
python3 parser.py release\(20250520\)/release/design7.v training_data/design7.csv
python3 parser.py release\(20250520\)/release/design8.v training_data/design8.csv
python3 parser.py release\(20250520\)/release/design9.v training_data/design9.csv
python3 parser.py release\(20250522\)/release2/trojan_design/design10.v training_data/design10.csv
python3 parser.py release\(20250522\)/release2/trojan_design/design11.v training_data/design11.csv
python3 parser.py release\(20250522\)/release2/trojan_design/design12.v training_data/design12.csv
python3 parser.py release\(20250522\)/release2/trojan_design/design13.v training_data/design13.csv
python3 parser.py release\(20250522\)/release2/trojan_design/design14.v training_data/design14.csv
python3 parser.py release\(20250522\)/release2/trojan_design/design15.v training_data/design15.csv
python3 parser.py release\(20250522\)/release2/trojan_design/design16.v training_data/design16.csv
python3 parser.py release\(20250522\)/release2/trojan_design/design17.v training_data/design17.csv
python3 parser.py release\(20250522\)/release2/trojan_design/design18.v training_data/design18.csv
python3 parser.py release\(20250522\)/release2/trojan_design/design19.v training_data/design19.csv
python3 parser.py release\(20250522\)/release2/trojan_free/design20.v training_data/design20.csv
python3 parser.py release\(20250522\)/release2/trojan_free/design21.v training_data/design21.csv
python3 parser.py release\(20250522\)/release2/trojan_free/design22.v training_data/design22.csv
python3 parser.py release\(20250522\)/release2/trojan_free/design23.v training_data/design23.csv
python3 parser.py release\(20250522\)/release2/trojan_free/design24.v training_data/design24.csv
python3 parser.py release\(20250522\)/release2/trojan_free/design25.v training_data/design25.csv
python3 parser.py release\(20250522\)/release2/trojan_free/design26.v training_data/design26.csv
python3 parser.py release\(20250522\)/release2/trojan_free/design27.v training_data/design27.csv
python3 parser.py release\(20250522\)/release2/trojan_free/design28.v training_data/design28.csv
python3 parser.py release\(20250522\)/release2/trojan_free/design29.v training_data/design29.csv

# This script labels the parsed .csv files using the results from the reference directory.
python3 label.py reference/reference/result0.txt training_data/design0.csv training_data_w_label/design0_label.csv
python3 label.py reference/reference/result1.txt training_data/design1.csv training_data_w_label/design1_label.csv
python3 label.py reference/reference/result2.txt training_data/design2.csv training_data_w_label/design2_label.csv
python3 label.py reference/reference/result3.txt training_data/design3.csv training_data_w_label/design3_label.csv
python3 label.py reference/reference/result4.txt training_data/design4.csv training_data_w_label/design4_label.csv
python3 label.py reference/reference/result5.txt training_data/design5.csv training_data_w_label/design5_label.csv
python3 label.py reference/reference/result6.txt training_data/design6.csv training_data_w_label/design6_label.csv
python3 label.py reference/reference/result7.txt training_data/design7.csv training_data_w_label/design7_label.csv
python3 label.py reference/reference/result8.txt training_data/design8.csv training_data_w_label/design8_label.csv
python3 label.py reference/reference/result9.txt training_data/design9.csv training_data_w_label/design9_label.csv
python3 label.py reference/reference/result10.txt training_data/design10.csv training_data_w_label/design10_label.csv
python3 label.py reference/reference/result11.txt training_data/design11.csv training_data_w_label/design11_label.csv
python3 label.py reference/reference/result12.txt training_data/design12.csv training_data_w_label/design12_label.csv
python3 label.py reference/reference/result13.txt training_data/design13.csv training_data_w_label/design13_label.csv
python3 label.py reference/reference/result14.txt training_data/design14.csv training_data_w_label/design14_label.csv
python3 label.py reference/reference/result15.txt training_data/design15.csv training_data_w_label/design15_label.csv
python3 label.py reference/reference/result16.txt training_data/design16.csv training_data_w_label/design16_label.csv
python3 label.py reference/reference/result17.txt training_data/design17.csv training_data_w_label/design17_label.csv
python3 label.py reference/reference/result18.txt training_data/design18.csv training_data_w_label/design18_label.csv
python3 label.py reference/reference/result19.txt training_data/design19.csv training_data_w_label/design19_label.csv
python3 label.py reference/reference/result20.txt training_data/design20.csv training_data_w_label/design20_label.csv
python3 label.py reference/reference/result21.txt training_data/design21.csv training_data_w_label/design21_label.csv
python3 label.py reference/reference/result22.txt training_data/design22.csv training_data_w_label/design22_label.csv
python3 label.py reference/reference/result23.txt training_data/design23.csv training_data_w_label/design23_label.csv
python3 label.py reference/reference/result24.txt training_data/design24.csv training_data_w_label/design24_label.csv
python3 label.py reference/reference/result25.txt training_data/design25.csv training_data_w_label/design25_label.csv
python3 label.py reference/reference/result26.txt training_data/design26.csv training_data_w_label/design26_label.csv
python3 label.py reference/reference/result27.txt training_data/design27.csv training_data_w_label/design27_label.csv
python3 label.py reference/reference/result28.txt training_data/design28.csv training_data_w_label/design28_label.csv
python3 label.py reference/reference/result29.txt training_data/design29.csv training_data_w_label/design29_label.csv