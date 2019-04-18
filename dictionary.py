class Dictionary:
    def __init__(self):
        self.list = []

    def add_word(self, (word, ispos)):
        if ispos:
            self.list.append((word, 1))
        else:
            self.list.append((word, -1))
    
    def find_word(self, word):
        start = 0
        end = len(self.list) - 1
        while True:
            if (end < start):
                return 0
            tmp = (start + end) / 2
            if self.list[tmp][0] == word:
                return self.list[tmp][1]
            if word < self.list[tmp][0]:
                end = tmp - 1
            else:
                start = tmp + 1

#d = Dictionary()
#d.add_word(("alpha", True))
#d.add_word(("beta", False))
#d.add_word(("charlie", False))
#d.add_word(("doctor", True))
#
#print d.list
#print d.find_word("alpha")
#print d.find_word("beta")
#print d.find_word("charlie")
#print d.find_word("doctor")
#
#print d.find_word("aa")
#print d.find_word("betac")
#print d.find_word("zebra")


