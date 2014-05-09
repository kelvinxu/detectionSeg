function [ boykov_acc ] = tune_pairwise (unary, img, labels )

for i = 75
for j = 1
    predictions_bk = predict_boykov(unary,img,j*.1,i);
    mean(seg_accuracy(predictions_bk, labels))
    
    
end
end

end

