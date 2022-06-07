setosa_correct_predictions = 13
setosa_incorrect_predictions = 0
total_setosa_prediction = 13
total_setosa_instances = 13

versicolor_correct_predictions = 15
versicolor_incorrect_predictions = 3
versicolor_total_predictions = 18
versicolor_total_instances = 16

virginica_correct_predictions = 6
virginica_incorrect_predictions = 1
virginica_total_predictions = 7
virginica_total_instances = 9

total_instances = total_setosa_instances + versicolor_total_instances + virginica_total_instances
check = virginica_total_predictions + versicolor_total_predictions + total_setosa_prediction
print(total_instances, check)

print("## setosa vs rest")

total_setosa_rest = versicolor_total_instances + virginica_total_instances
precision_setosa = setosa_correct_predictions/float(total_setosa_prediction)
print("precision_setosa: ", precision_setosa)

recall_setosa = setosa_correct_predictions/float(total_setosa_instances)
print("recall_setosa: ", recall_setosa)

weight_setosa = float(total_setosa_instances)/total_instances
print("weight_setosa: ", weight_setosa)

F1_setosa = 2*precision_setosa*recall_setosa/(precision_setosa+recall_setosa)
print("F1_setosa: ", F1_setosa)

weighted_F1_setosa = weight_setosa*F1_setosa
print("weighted F1_setosa: ", weighted_F1_setosa)

print("\n")
print("## versicolor vs rest")

total_versicolor_rest = versicolor_total_instances + virginica_total_instances
precision_versicolor = versicolor_correct_predictions/float(versicolor_total_predictions)
print("precision_versicolor: ", precision_versicolor)

recall_versicolor = versicolor_correct_predictions/float(versicolor_total_instances)
print("recall_versicolor: ", recall_versicolor)

weight_versicolor = float(versicolor_total_instances)/total_instances
print("weight_versicolor: ", weight_versicolor)

F1_versicolor = 2*precision_versicolor*recall_versicolor/(precision_versicolor+recall_versicolor)
print("F1_versicolor: ", F1_versicolor)

weighted_F1_versicolor = weight_versicolor*F1_versicolor
print("weighted F1_versicolor: ", weighted_F1_versicolor)

print("\n")
print("## virginica vs rest")

total_virginica_rest = virginica_total_instances + virginica_total_instances
precision_virginica = virginica_correct_predictions/float(virginica_total_predictions)
print("precision_virginica: ", precision_virginica)

recall_virginica = virginica_correct_predictions/float(virginica_total_instances)
print("recall_virginica: ", recall_virginica)

weight_virginica = float(virginica_total_instances)/total_instances
print("weight_virginica: ", weight_virginica)

F1_virginica = 2*precision_virginica*recall_virginica/(precision_virginica+recall_virginica)
print("F1_virginica: ", F1_virginica)


weighted_F1_virginica = weight_virginica*F1_virginica
print("weighted F1_virginica: ", weighted_F1_virginica)


print("\n")
macro_weighted_precision = weight_setosa*precision_setosa + weight_virginica*precision_virginica + weight_versicolor*precision_versicolor
print("macro weighted precision: ", macro_weighted_precision)

macro_weighted_recall = weight_setosa*recall_setosa + weight_virginica*recall_virginica + weight_versicolor*recall_versicolor
print("macro weighted recall: ", macro_weighted_recall)

print("\n")
average_precision = (1/3.0) * (precision_setosa + precision_virginica  + precision_versicolor)
print("average precision: ", average_precision)

average_recall = (1/3.0) * (recall_versicolor + recall_virginica + recall_setosa)
print("average recall: ", average_recall)

print("\n")
print("macro weighted PR harmonic mean F1: ", 2.0*macro_weighted_precision*macro_weighted_recall/(macro_weighted_precision+macro_weighted_recall))
print("average PR harmonic mean F1: ", 2.0*(average_recall*average_precision)/(average_recall+average_precision))
print("1 vs rest F1: ", weighted_F1_virginica + weighted_F1_setosa + weighted_F1_versicolor)
