import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def predict(text):
    # トークナイザの取得
    tokenizer = AutoTokenizer.from_pretrained('./AI_model')
    
    # モデルの取得
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (AutoModelForSequenceClassification
        .from_pretrained('./AI_model')
        .to(device))
    
    #入力データ
    sample_text = text
    
    inputs = tokenizer(sample_text, return_tensors="pt")
    labels =['1:very low', '2:low', '3:normal','4:high','5:very high']
    model.eval()

    with torch.no_grad():
        outputs = model(inputs["input_ids"].to(device), inputs["attention_mask"].to(device))
        prediction = torch.nn.functional.softmax(outputs.logits,dim=1)
    
    #ラベル名を表示
    pred_label = labels[int(torch.argmax(prediction))]
    
    return pred_label