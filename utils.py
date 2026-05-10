import os
import yaml
import zipfile


def counting(data):
    counts = {}
    for item in data:
        sentRaw = item['raw']
        sentGold = item['norm']
        for wordRaw, wordGold in zip(sentRaw, sentGold):
            if wordRaw not in counts:
                counts[wordRaw] = {}
            if wordGold not in counts[wordRaw]:
                counts[wordRaw][wordGold] = 0
            counts[wordRaw][wordGold] += 1
    return counts

def mfr(input_sent, counts):
    predictions = []
    for word in input_sent:
        if word in counts:
            replacement = max(counts[word], key=counts[word].get)
        else:
            replacement = word
        predictions.append(replacement)
    return predictions

def evaluate(raw, gold, pred, ignCaps=False, verbose=False, info=True):
    cor = 0
    changed = 0
    total = 0

    # 1. 에러 출력 방식을 print나 raise로 변경
    if len(gold) != len(pred):
        print(f'Error: gold normalization contains a different number of sentences({len(gold)}) compared to system output({len(pred)})')
        return None

    for sentRaw, sentGold, sentPred in zip(raw, gold, pred):
        if len(sentGold) != len(sentPred):
            print('Error: a sentence has a different length in your output, check the order of the sentences')
            continue # 혹은 break
            
        for wordRaw, wordGold, wordPred in zip(sentRaw, sentGold, sentPred):
            if ignCaps:
                wordRaw, wordGold, wordPred = wordRaw.lower(), wordGold.lower(), wordPred.lower()
            
            if wordRaw != wordGold:
                changed += 1
            if wordGold == wordPred:
                cor += 1
            elif verbose:
                print(wordRaw, wordGold, wordPred)
            total += 1

    accuracy = cor / total
    lai = (total - changed) / total
    
    # 2. 변수명을 err_rate 등으로 변경하여 함수 호출과 혼동 피하기
    err_rate = (accuracy - lai) / (1 - lai) if (1 - lai) != 0 else 0

    if info:
        print('Baseline acc.(LAI): {:.2f}'.format(lai * 100)) 
        print('Accuracy:           {:.2f}'.format(accuracy * 100)) 
        print('ERR:                {:.2f}'.format(err_rate * 100))

    return lai, accuracy, err_rate



def zip_files_flat(source_dir, output_zip, flag=None):
    """
    Zips files from the source directory. Supports flat structure with '-j' flag.

    Args:
        source_dir (str): Path to the directory containing files to be zipped.
        output_zip (str): Path to the output zip file.
        flag (str, optional): Use '-j' to zip files without directory structure.
    """
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if flag == '-j':
                    # Add file without directory structure
                    zipf.write(file_path, arcname=file)
                else:
                    # Preserve directory structure
                    zipf.write(file_path, arcname=os.path.relpath(file_path, source_dir))
    print(f"Created zip file: {output_zip}")
