words = open("words.txt", 'r').read().splitlines()

wordsToNums = {}
numsToWords = {}
i = 0
for word in words:
    print(word, word.lower())
    word = word.lower()
    wordsToNums[word] = i
    numsToWords[i] = word
    i += 1

out = open("dicts.js", 'w')

out.write("""
export default {{
    "wordsToNums": {0},
    "numsToWords": {1}
}}
""".format(wordsToNums, numsToWords))

out.close()