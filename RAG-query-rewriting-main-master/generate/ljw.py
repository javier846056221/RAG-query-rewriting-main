import json
from evaluation import eval_question_answering


evalfile=r'D:\workcode\RAG-query-rewriting-main-master\generate\outputs\onlyq-hotpot\onlyq-hotpot-metrics-pt.jsonl'
outputfile=r'D:\workcode\RAG-query-rewriting-main-master\generate\outputs\onlyq-hotpot\hotpot-dev-p3-pt.jsonl'
with open(evalfile, 'a', encoding='utf8') as evalout:
    emscore, length, f1 = eval_question_answering(outputfile, None)
    outmetrics = {
        'outputfile': outputfile,
        'exact match': emscore,
        'F1': f1,
        'length': length,

    }
    print(f'Exact Match: {emscore}; F1: {f1}; Avg.Length: {length}')
    evalout.write(json.dumps(outmetrics) + '\n')