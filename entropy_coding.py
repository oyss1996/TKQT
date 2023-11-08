import numpy as np


def run_length_encoding(data):
    # data += 65  # [0, 32) -> [65, 97) -> [A, `]
    data += 63  # [0, 64) -> [63, 127) -> [?,@,A_Z,...,a_z,..]
    rle = []
    count = 1
    for i in range(1, len(data)):
        if data[i] == data[i - 1]:
            count += 1
        else:
            if count >= 3:
                rle += [chr(data[i - 1]), str(count - 3)]
            else:
                rle += [chr(data[i - 1])] * count
            count = 1
    if count >= 3:
        rle += [chr(data[-1]), str(count - 3)]
    else:
        rle += [chr(data[-1])] * count
    data -= 63
    return ''.join(rle)


def run_length_decoding(code):
    data = []
    i = 0
    while i < len(code):
        char = code[i]
        if '0' <= char <= '9':
            while i < len(code)-1 and '0' <= code[i+1] <= '9':
                char += code[i+1]
                i += 1
            count = int(char) + 2
            data += [data[-1]] * count
        else:
            data.append(ord(char)-63)
        i += 1
    return np.asarray(data)


def run_length_encoding_bit(data):      # bit码流的游程编码
    rle = []
    if data[0] == '0': rle.append('0')   # 如果第一个bit不是1的话，则记下0，表示有0个1
    count = 1
    for i in range(1, len(data)):
        if data[i] == data[i - 1]:
            count += 1
        else:
            rle.append(str(count))
            count = 1
    rle.append(str(count))
    return ' '.join(rle)




"""
    Huffman code for collection of strings.
    Uses heapq for structure
    Author: George Heineman
"""
import heapq


class Node:
    def __init__(self, prob, symbol=None):
        """Create node for given symbol and probability."""
        self.left = None
        self.right = None
        self.symbol = symbol
        self.prob = prob

    # Need comparator method at a minimum to work with heapq
    def __lt__(self, other):
        return self.prob < other.prob

    def encode(self, encoding):
        """Return bit encoding in traversal."""
        if self.left is None and self.right is None:
            yield self.symbol, encoding
        else:
            for v in self.left.encode(encoding + '0'):
                yield v
            for v in self.right.encode(encoding + '1'):
                yield v


class Huffman:
    def __init__(self, initial):
        """Construct encoding given initial corpus."""
        self.initial = initial

        # Count frequencies
        freq = {}
        for _ in initial:
            if _ in freq:
                freq[_] += 1
            else:
                freq[_] = 1

        # Construct priority queue
        pq = []
        for symbol in freq:
            pq.append(Node(freq[symbol], symbol))
        heapq.heapify(pq)

        # special case: what if only one symbol?
        if len(pq) == 1:
            self.root = Node(1)
            self.root.left = pq[0]
            self.encoding = {symbol: '0'}
            return

        # Huffman Encoding Algorithm
        while len(pq) > 1:
            n1 = heapq.heappop(pq)
            n2 = heapq.heappop(pq)
            n3 = Node(n1.prob + n2.prob)
            n3.left = n1
            n3.right = n2
            heapq.heappush(pq, n3)

        # Record
        self.root = pq[0]
        self.encoding = {}
        for sym, code in pq[0].encode(''):
            self.encoding[sym] = code

    def __repr__(self):
        """Show encoding"""
        return 'huffman:' + str(self.encoding)

    def encode(self, s):
        """Return bit string for encoding."""
        bits = ''
        for _ in s:
            if not _ in self.encoding:
                raise ValueError("'" + _ + "' is not encoded character")
            bits += self.encoding[_]
        return bits

    def decode(self, bits):
        """Decode ASCII bit string for simplicity."""
        node = self.root
        s = ''
        for _ in bits:
            if _ == '0':
                node = node.left
            else:
                node = node.right

            if node.symbol:
                s += node.symbol
                node = self.root

        return s

