import argparse
import os
import json

from inference import run_main, run_searchre
from evaluation import (
    eval_recall,
    eval_question_answering,
    eval_fact_checking,
    eval_dialogue_system
)
import wikienv
from bingenv import BingEnv
from tqdm import tqdm

def datapath(dataset, split):
    if dataset == 'tqa-eg':
        inputfile = '/xinbei_data/replug/tqa/egs/tqa-test-eg.jsonl'
    elif dataset == 'tqa':
        inputfile = f'/xinbei_data/replug/tqa/tqa-{split}.jsonl'
    elif dataset == 'hotpot':
        inputfile = rf'D:\workcode\RAG-query-rewriting-main-master\datasets\\tasks\data\hotpot_{split}_v1_simplified.json'
    elif dataset == 'nq':
        inputfile = f'/xinbei_data/replug/nq/nq-{split}.jsonl'
    elif dataset == 'tqa-wrong':
        inputfile = f'/xinbei_data/replug/generate/forRL/rewrite-tqa/wrongcases.jsonl'
    elif dataset == 'tqa-totalwrong':
        inputfile = f'/xinbei_data/replug/generate/forRL/rewrite-tqa/totalwrongcases.jsonl'
    elif dataset == 'tqa-filter':
        inputfile = f'/xinbei_data/replug/generate/filtered-nq/tqa-{split}-filtered.jsonl'
    elif dataset == 'nq-filter':
        inputfile = f'/xinbei_data/replug/generate/filtered-nq/nq-{split}-filtered.jsonl' 
    elif dataset == 'webq':
        inputfile = f'/xinbei_data/replug/webq/webq-{split}.jsonl'
    elif dataset == 'amb':
        inputfile = f'/xinbei_data/replug/ambignq/{split}.jsonl'
    elif dataset == 'fbqa':
        inputfile = f'/xinbei_data/replug/FreebaseQA-master/{split}.jsonl'
    elif dataset == 'truqa':
        inputfile = f'/xinbei_data/replug/truthqa/truthqa{split}.jsonl'
    elif dataset == 'gaokao-geo':
        inputfile = "/xinbei_data/replug/generate/AGIEval/data/qav/gaokao-geography.jsonl"
    elif dataset == 'popqa':
        inputfile =  f'/xinbei_data/replug/generate/popqa/{split}.jsonl'
    return inputfile

def readfiles(infile):

    if infile.endswith('json'): 
        lines = json.load(open(infile, 'r', encoding='utf8'))
    elif infile.endswith('jsonl'): 
        lines = open(infile, 'r', encoding='utf8').readlines()
        lines = [json.loads(l, strict=False) for l in lines]
    else:
        raise NotImplementedError
    if len(lines) == 0:
        return []
    if len(lines[0]) == 1 and lines[0].get('prompt'): 
        lines = lines[1:] ## skip prompt line
    if 'answer' in lines[0].keys() and type(lines[0]['answer']) == str:
        for l in lines:
            l['answer'] = [l['answer']]
    return lines

def bing_bl(args, datatype, max_tokens, prompt):
    inputfile = datapath(args.dataset, args.split)
    inlines = readfiles(inputfile)
    if (args.temp is None) or (args.temp == 0):
        outputfolder = f'{args.output_dir}/bingbl-{args.dataset}'
    else: # tempature > 0
        outputfolder = f'{args.output_dir}/bingbl(n={args.n},temp={args.temp})-{args.engine}/{args.dataset}'
    os.makedirs(outputfolder, exist_ok=True)
    outputfile = f'{outputfolder}/{args.dataset}-{args.split}-p{args.pid}-{args.post}.jsonl'
    # search results
    sre = f'{outputfolder}/{args.search}-{args.dataset}-searchresult-{args.post}.jsonl'
    begin = 0
    if os.path.exists(sre): 
        print(f"Loading existing search results {sre}")
        inlines_w_sre = readfiles(sre)
        begin = len(inlines_w_sre)
        inlines[:len(inlines_w_sre)] = inlines_w_sre
    if args.nums:
        inlines = inlines[:args.nums]
    # search on bing
    fsre = open(sre, 'a')
    nowiki = 0
    if args.search == 'wiki':
        mywikienv = wikienv.WikiEnv()
    elif args.search == 'bing':
        mybingenv = BingEnv()
    for il in tqdm(range(begin, len(inlines))):
        query = 'search[' + inlines[il]['question'] +']'
        if args.search=='bing':
            # obs, reward, done, info = env.step(query, args.use_en, func='bm25', gold=inlines[il]['answer'])
            obs, reward, done, info = mybingenv.step(query, args.use_en, func='plain', gold=inlines[il]['answer'])
        else: 
            obs, reward, done, info = mywikienv.step(query)
        # obs, reward, done, info = env.step(query)
        inlines[il]['output'] = obs
        if args.sele == True and (obs.startswith("Similar:") or obs.strip == ''):
            inlines[il]['output'] = ''
            nowiki += 1
        fsre = open(sre, 'a')
        fsre.write(json.dumps(inlines[il])+'\n')
        fsre.close()
    # prompt gpt
    run_main(inlines, outputfile, args.engine, prompt, max_tokens, args.n, args.temp, args.endwith)
    # eval J
    evalfile = f'{outputfolder}/{args.search}-{args.dataset}-metrics-{args.post}.jsonl'
    with open(evalfile, 'a') as evalout:
        emscore, length, f1 = eval_question_answering(outputfile, args.endwith)
        outmetrics = {
            'outputfile': outputfile,
            'prompt': prompt,
            'exact match': emscore,
            'F1': f1,
            'length': length,
            'nowiki': nowiki,
            "search": args.search,
            'nums':args.nums
        }
        print(f'Exact Match: {emscore}; F1: {f1}; Avg.Length: {length}: No wiki: {nowiki}')
        evalout.write(json.dumps(outmetrics) + '\n')


def step1(args, datatype, max_tokens, prompt):
    inputfile = datapath(args.dataset, args.split)
    inlines = readfiles(inputfile)
    if args.nums:
        inlines = inlines[:args.nums]
    if (args.temp is None) or (args.temp == 0):
        outputfolder = f'{args.output_dir}/onlyq-{args.dataset}'
    else: # tempature > 0
        outputfolder = f'{args.output_dir}/onlyq(n={args.n},temp={args.temp})-{args.engine}/{args.dataset}'
    os.makedirs(outputfolder, exist_ok=True)
    outputfile = f'{outputfolder}/{args.dataset}-{args.split}-p{args.pid}-{args.post}.jsonl'
    
    run_main(inlines, outputfile, args.engine, prompt, max_tokens, args.n, args.temp, args.endwith)
    # eval 
    evalfile = f'{outputfolder}/onlyq-{args.dataset}-metrics-{args.post}.jsonl'
    with open(evalfile, 'a', encoding='utf8') as evalout:
        emscore, length, f1 = eval_question_answering(outputfile, args.endwith)
        outmetrics = {
            'outputfile': outputfile,
            'prompt': prompt,
            'exact match': emscore,
            'F1': f1,
            'length': length,
            "search": args.search,
            'nums':args.nums
        }
        print(f'Exact Match: {emscore}; F1: {f1}; Avg.Length: {length}')
        evalout.write(json.dumps(outmetrics) + '\n')

def wiki(args, datatype, max_tokens, prompt):
    inputfile = datapath(args.dataset, args.split)
    inlines = readfiles(inputfile)
    if args.nums:
        inlines = inlines[:args.nums]
    if (args.temp is None) or (args.temp == 0):
        outputfolder = f'{args.output_dir}/searchQ-{args.dataset}'
    else: # tempature > 0
        outputfolder = f'{args.output_dir}/searchQwiki(n={args.n},temp={args.temp})-{args.engine}/{args.dataset}'
    os.makedirs(outputfolder, exist_ok=True)
    outputfile = f'{outputfolder}/{args.search}-{args.dataset}-{args.split}-p{args.pid}-{args.post}.jsonl'
    # search in wikienv 
    nowiki = 0
    sre = f'{outputfolder}/{args.search}-{args.dataset}-searchresult-{args.post}.jsonl'
    begin = 0
    if os.path.exists(sre): 
        print(f"Loading existing search results {sre}")
        inlines_w_sre = readfiles(sre)
        begin = len(inlines_w_sre)
        inlines[:len(inlines_w_sre)] = inlines_w_sre
        if args.nums:
            inlines = inlines[:args.nums]
    
    fsre = open(sre, 'a')
    if args.search == 'wiki':
        env = wikienv.WikiEnv()
    elif args.search == 'bing':
        env = BingEnv()
    for il in tqdm(range(begin, len(inlines))):
        query = 'search[' + inlines[il]['question'] +']'
        if args.search=='bing':
            obs, reward, done, info = env.step(query, args.use_en)
        else: 
            obs, reward, done, info = env.step(query)
        # obs, reward, done, info = env.step(query)
        inlines[il]['output'] = obs
        if args.sele == True and (obs.startswith("Similar:") or obs.strip == ''):
            inlines[il]['output'] = ''
            nowiki += 1
        fsre.write(json.dumps(inlines[il])+'\n')
        # with open(sre, 'w') as f:
        #     for line in inlines:
        #         f.write(json.dumps(line)+'\n')      

    run_main(inlines, outputfile, args.engine, prompt, max_tokens, args.n, args.temp, args.endwith)
    # eval 
    evalfile = f'{outputfolder}/{args.search}-{args.dataset}-metrics-{args.post}.jsonl'
    with open(evalfile, 'a', encoding='utf8') as evalout:
        emscore, length, f1 = eval_question_answering(outputfile, args.endwith)
        outmetrics = {
            'outputfile': outputfile,
            'prompt': prompt,
            'exact match': emscore,
            'F1': f1,
            'length': length,
            'nowiki': nowiki,
            "search": args.search,
            'nums':args.nums
        }
        print(f'Exact Match: {emscore}; F1: {f1}; Avg.Length: {length}: No wiki: {nowiki}')
        evalout.write(json.dumps(outmetrics,ensure_ascii=False) + '\n')

def searchrewrite(args, datatype, max_tokens, prompt):
    inputfile = datapath(args.dataset, args.split)
    inlines = readfiles(inputfile)
    if args.nums:
        inlines = inlines[:args.nums]
    if (args.temp is None) or (args.temp == 0):
        outputfolder = f'{args.output_dir}/searchre-{args.dataset}'
    else: # tempature > 0
        outputfolder = f'{args.output_dir}/searchre(n={args.n},temp={args.temp})-{args.engine}/{args.dataset}'
    os.makedirs(outputfolder, exist_ok=True)
    outputfile = f'{outputfolder}/{args.search}-{args.dataset}-{args.split}-p{args.pid}-{args.post}.jsonl'
    # search in wikienv 
    nowiki = 0
    sre = f'{outputfolder}/{args.search}-{args.dataset}-searchresult-{args.post}.jsonl'
    if os.path.exists(sre): 
        print(f"Loading existing search results {sre}")
        inlines = readfiles(sre)
        if args.nums:
            inlines = inlines[:args.nums]
    else:
        if args.search == 'wiki':
            env = wikienv.WikiEnv()
        elif args.search == 'bing':
            env = BingEnv()
        for il in tqdm(range(len(inlines))):
            query = 'search[' + inlines[il]['question'] +']'
            if args.search=='bing':
                obs, reward, done, info = env.step(query, args.use_en)
            else: 
                obs, reward, done, info = env.step(query)
            # obs, reward, done, info = env.step(query)
            inlines[il]['output'] = obs
            if args.sele == True and (obs.startswith("Similar:") or obs.strip == ''):
                inlines[il]['output'] = ''
                nowiki += 1
        with open(sre, 'w') as f:
            for line in inlines:
                f.write(json.dumps(line)+'\n') 
    # rewrite the search results
    outsr = f'{outputfolder}/{args.search}-{args.dataset}-searchrewrite-{args.post}.jsonl'
    if not os.path.exists(outsr):
        prompt_rew = "Summarize the following passage for this question, end with '**'. \n\n Question: {query} Passage: {output} \n\n Summary:"
        run_searchre(inlines, outsr, args.engine, prompt_rew, max_tokens, args.n, args.temp, args.endwith)
    rewrited = readfiles(outsr)
    for i in range(len(inlines)):
        inlines[i]['output'] = rewrited[i]['output']
    # prompt_rew = "Summarize the following passage for this question, end with '**'. \n\n Question: {query} Passage: {output} \n\n Summary:"
    # rewrited = run_searchre(inlines, outsr, args.engine, prompt_rew, max_tokens, args.n, args.temp, args.endwith)
    # for i in range(len(inlines)):
    #     inlines[i]['output'] = rewrited[i]
    run_main(inlines, outputfile, args.engine, prompt, max_tokens, args.n, args.temp, args.endwith)
    # eval 
    evalfile = f'{outputfolder}/{args.search}-{args.dataset}-metrics-{args.post}.jsonl'
    with open(evalfile, 'a') as evalout:
        emscore, length, f1 = eval_question_answering(outputfile, args.endwith)
        outmetrics = {
            'outputfile': outputfile,
            'prompt': prompt,
            'exact match': emscore,
            'F1': f1,
            'length': length,
            'nowiki': nowiki,
            "search": args.search,
            'nums':args.nums
        }
        print(f'Exact Match: {emscore}; F1: {f1}; Avg.Length: {length}: No wiki: {nowiki}')
        evalout.write(json.dumps(outmetrics) + '\n')

def rewrite(args, datatype, max_tokens, prompt):
    inputfile = datapath(args.dataset, args.split)
    inlines = readfiles(inputfile)
    if args.nums:
        inlines = inlines[:args.nums]
    if (args.temp is None) or (args.temp == 0):
        outputfolder = f'{args.output_dir}/rewrite-{args.dataset}'
    else: # tempature > 0
        outputfolder = f'{args.output_dir}/rewrite(n={args.n},temp={args.temp})-{args.engine}/{args.dataset}'
    os.makedirs(outputfolder, exist_ok=True)
    outputfile = f'{outputfolder}/{args.search}-{args.dataset}-{args.split}-rew-p{args.pid}-{args.post}.jsonl'
    # inlines = inlines[:100]
    if os.path.exists(outputfile):
        print(f"Loading existing rewritten {outputfile}")
        inlines = readfiles(outputfile)
    else:
        run_main(inlines, outputfile, args.engine, prompt, max_tokens, args.n, args.temp, args.endwith)
        inlines = readfiles(outputfile)
    if args.nums:
        inlines = inlines[:args.nums]
    # search
    sre = f'{outputfolder}/{args.search}-{args.dataset}-searchresult-{args.pid}-{args.post}.jsonl' if not args.max_obs else f'{outputfolder}/{args.search}-{args.dataset}-searchresult-{args.max_obs}-{args.pid}-{args.post}.jsonl'
    if os.path.exists(sre):
        print(f"Loading existing search results {sre}")
        srelines = readfiles(sre)
        if len(srelines) < len(inlines):
            print('continue from ', len(srelines))
            inlines = inlines[len(srelines):]
        else:
            return
    if args.search == 'wiki':
        env = wikienv.WikiEnv()
    elif args.search == 'bing':
        env = BingEnv()
    env.appendsimilar = True
    nowiki = 0
    sref = open(sre, 'a')
    for il in tqdm(range(len(inlines))):
        if args.think:
            # print(inlines[il])
            if "Query:" not in inlines[il]['output'][0]:
                qs= ['']
                inlines[il]['thought'] = inlines[il]['output'][0]
            else:
                qs = inlines[il]['output'][0].split("Query:")[1].split(";")
                inlines[il]['thought'] = inlines[il]['output'][0].split("Query:")[0]
        else:
            qs = inlines[il]['output'][0].split(";")
        inlines[il]['output'] = []
        for q in qs:
            print(q)
            query = 'search[' + q +']'
            if args.search=='bing':
                obs, reward, done, info = env.step(query, args.use_en, func='plain', gold='')
            else: 
                obs, reward, done, info = env.step(query)
            if args.sele == True and obs.startswith("Similar:"):
                inlines[il]['output'].append('')
                nowiki += 1
            else:
                inlines[il]['output'].append(obs)
                # print(obs)
        if args.max_obs:
            for o in range(len(inlines[il]['output'])):
                # word
                # print(inlines[il]['output'][o])
                trunc = " ".join(inlines[il]['output'][o].split(" ")[:args.max_obs])
                inlines[il]['output'][o] = trunc
        inlines[il]['output'] = [" ".join(inlines[il]['output'])]
        sref.write(json.dumps(inlines[il])+'\n')
        # os._exit()
    # with open(sre, 'a') as f:
    #     for line in inlines:
    #         f.write(json.dumps(line)+'\n')

def rewrite2(args, datatype, max_tokens, prompt):
    searchre = True
    # for hotpot
    if (args.temp is None) or (args.temp == 0):
        outputfolder = f'{args.output_dir}/rewrite-{args.dataset}'
    else: # tempature > 0
        outputfolder = f'{args.output_dir}/rewrite(n={args.n},temp={args.temp})-{args.engine}/{args.dataset}'
    sre = f'{outputfolder}/{args.search}-{args.dataset}-searchresult-{args.repid}-{args.post}.jsonl' if not args.max_obs else f'{outputfolder}/{args.search}-{args.dataset}-searchresult-{args.max_obs}-{args.repid}-{args.post}.jsonl'
    inlines = readfiles(sre)
    if args.nums:
        inlines = inlines[:args.nums]
    if (args.temp is None) or (args.temp == 0):
        outputfolder = f'{args.output_dir}/rewrite2con-{args.dataset}'
    else: # tempature > 0
        outputfolder = f'{args.output_dir}/rewrite2con(n={args.n},temp={args.temp})-{args.engine}/{args.dataset}'
    os.makedirs(outputfolder, exist_ok=True)
    outputfile = f'{outputfolder}/{args.search}-{args.dataset}-{args.split}-rew2con-p{args.pid}-{args.post}-{args.repid}.jsonl' if not args.max_obs else f'{outputfolder}/{args.search}-{args.dataset}-{args.split}-rew2con-p{args.pid}-{args.post}-{args.max_obs}-{args.repid}.jsonl'
    
    if searchre == True:
        outsr = f'{outputfolder}/{args.search}-{args.dataset}-searchrewrite-{args.post}.jsonl'
        if not os.path.exists(outsr):
            prompt_rew = "Summarize the following passage for this question, end with '**'. \n\n Question: {query} Passage: {output} \n\n Summary:"
            run_searchre(inlines, outsr, args.engine, prompt_rew, max_tokens, args.n, args.temp, args.endwith)
        rewrited = readfiles(outsr)
        for i in range(len(inlines)):
            inlines[i]['output'] = rewrited[i]['output']
    run_main(inlines, outputfile, args.engine, prompt, max_tokens, args.n, args.temp, args.endwith)
    # eval 
    evalfile = f'{outputfolder}/{args.search}-rewrite2con-{args.dataset}-metrics-{args.repid}-{args.post}.jsonl' if not args.max_obs else f'{outputfolder}/{args.search}-rewrite2con-{args.dataset}-{args.post}-metrics-{args.max_obs}-{args.repid}.jsonl'
    with open(evalfile, 'a') as evalout:
        emscore, length, f1 = eval_question_answering(outputfile, args.endwith)
        outmetrics = {
            'outputfile': outputfile,
            'prompt': prompt,
            'exact match': emscore,
            'F1': f1,
            'length': length,
            "search": args.search,
            'nums':args.nums
        }
        print(f'Exact Match: {emscore}; F1: {f1}; Avg.Length: {length}')
        evalout.write(json.dumps(outmetrics) + '\n')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    # Required parameters
    parser.add_argument("--dataset", default=None, type=str, required=True,
        help="dataset name: [nq, tqa, webq, wizard, fever, fm2]",
    )
    parser.add_argument("--task", default='step1', type=str, required=True,
        help="task name: [step1, wiki, rewrite, rewrite2], should be either 1 or 2",
    )
    parser.add_argument("--split", default='test', type=str, required=True,
        help="dataset split: [train, dev, test]",
    )
    parser.add_argument("--engine", default='text-davinci-002', type=str, required=False,
        help="text-davinci-002 (used in our experiments), code-davinci-002",
    )
    parser.add_argument("--n", default=1, type=int, required=False, help='--num_sequence')
    parser.add_argument("--temp", default=0, type=float, required=False, help='--temperature')
    parser.add_argument('--pid', default=1, type=int, required=True)
    parser.add_argument('--endwith', default=None, type=str)
    parser.add_argument('--sele', action='store_true')
    parser.add_argument('--promptfile', default='myprompt', type=str)
    parser.add_argument('--search', type=str, default='wiki')
    parser.add_argument('--nums', type=int, default=None)
    parser.add_argument('--post', type=str, default='pt', help='postfix')
    parser.add_argument('--max_obs', type=int, default=None)
    parser.add_argument('--repid', type=int, default=None)
    parser.add_argument('--think', action='store_true')
    parser.add_argument('--output_dir',type=str, default='./outputs')
    parser.add_argument('--use_en', action="store_true")

    args = parser.parse_args()

    args.endwith = '**'
    # args.endwith = None
    # if args.dataset in ['nq', 'webq', 'tqa', 'twiki','tqa-eg', 'hotpot','tqa-wrong','tqa-totalwrong','tqa-filter', 'nq-filter','webq','amb','fbqa','truqa','gaokao']:
    datatype = 'question answering'
    # elif args.dataset in ['fever', 'fm2']:
    #     datatype = 'fact checking'
    # elif args.dataset in ['wizard']: 
    #     datatype = 'dialogue system'
    # else: # other task type?
    #     raise NotImplementedError

    # if args.task == 'step1':
    #     max_tokens = 300
    # elif args.task == 'step2':
    #     if datatype == 'dialogue system':
    #         max_tokens = 50
    #     else: # QA and Fact ...
    #         max_tokens = 10
    max_tokens = 300

    promptfile = args.promptfile
    promptlines = open(f'./inprompts/{promptfile}.jsonl', 'r',encoding='utf-8').readlines()

    for line in promptlines:
        line = json.loads(line)

        if line['type'] == datatype and line['task'] == args.task and line['pid'] == args.pid:
            prompt = line['prompt']
            pid = line['pid']

            if args.task == 'step1':
                outputs = step1(args, datatype, max_tokens, prompt)
            elif args.task == 'wiki':
                outputs = bing_bl(args, datatype, max_tokens, prompt)
            elif args.task == 'searchre':
                outputs = searchrewrite(args, datatype, max_tokens, prompt)
            elif args.task == 'rewrite':
                outputs = rewrite(args, datatype, max_tokens, prompt)
            elif args.task == 'rewrite2':
                outputs = rewrite2(args, datatype, max_tokens, prompt)
            else:  ## should be either 1 or 2
                raise NotImplementedError
            
            if promptfile == 'regular':
                break ## only use the first prompt
    print(parser)