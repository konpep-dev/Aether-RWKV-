import re, os
GREEK = re.compile(r'[Α-ωϊΐόύώήέ]')
out = []
for f in ['generate_dataset.py','inference.py','model.py','tokenizer.py','train.py']:
    p = os.path.join(r'C:\Users\konpep\Desktop\moled\github', f)
    with open(p, encoding='utf-8', errors='replace') as fh:
        lines = fh.readlines()
    greek_lines = [(i+1, line.rstrip()) for i, line in enumerate(lines) if GREEK.search(line)]
    out.append(f'=== {f}: {len(greek_lines)} Greek lines ===')
    for ln, txt in greek_lines:
        out.append(f'  L{ln}: {txt}')
with open(r'C:\Users\konpep\AppData\Local\Temp\opencode\greek_lines.txt', 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(out))
print('written')
