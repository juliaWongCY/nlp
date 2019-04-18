To run sentiment analysis on the data set:
python vader_analysis.py <excel_file> <sheet_name>
eg. python vader_analysis Data\master.xlsx Sheet1

To run the decision tree:
python decision_tree.py <excel_file> 
**Pre-conditions:
The <excel_file> needs to have sheets with names Set1-Set5 
and sentiment scores are already generated.

To run the SVM:
python support_vector_machine.py <excel_file>
**Pre-conditions:
The <excel_file> needs to have sheets with names Set1-Set5 
and sentiment scores are already generated.