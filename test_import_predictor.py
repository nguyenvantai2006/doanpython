import sys
sys.path.insert(0, r'C:/Users/nhatd/Downloads/SentimentAnalysis-main/SentimentAnalysis-main')
try:
    from src.bert_inference import BERTPredictor
    print('Imported BERTPredictor')
    try:
        p = BERTPredictor()
        print('Loaded predictor key:', p.model_key)
        print('Explain sample:', p.explain_keywords('sản phẩm tuyệt vời giao hàng nhanh', top_n=3))
    except Exception as e:
        print('Predictor load/explain skipped or failed:', e)
except Exception as e:
    print('Import failed:', e)
