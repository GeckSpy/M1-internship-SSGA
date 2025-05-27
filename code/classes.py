import numpy as np


class UnionFind:
    def __init__(self, n):
        self.parent = np.arange(n)
        self.component:list[set] = [set([i]) for i in range(n)]
        self.count = n

    def find(self, x)->int:
        # Path compression
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    
    def union(self, x, y)->bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        
        if len(self.component[rx]) < len(self.component[ry]):
            rx, ry = ry, rx

        self.parent[ry] = rx
        self.component[rx] = self.component[rx].union(self.component[ry])
        self.component[ry].clear()
        self.count -= 1
        return True
    
    def __str__(self):
        return str(self.component)



class Pair:
    def __init__(self, first, second):
        self.first = first
        self.second = second

    def __str__(self):
        return "(" + str(self.first) + ", " + str(self.second) +")"
    
    def get(self):
        return self.first, self.second



class MinHeap:
    def __init__(self):
        self.array:list[Pair] = []

    def __str__(self):
        return str(self.array)

    def left(self, i):
        return 2*i+1
    
    def right(self, i):
        return 2*i+2

    def isEmpty(self):
        return self.array==[]

    def insert(self, elt, val):
        """Insert a new element into the Min Heap."""
        self.array.append(Pair(elt, val))
        i = len(self.array) - 1
        while i>0 and self.array[(i - 1) // 2].second > self.array[i].second:
            self.array[i], self.array[(i - 1) // 2] = self.array[(i - 1) // 2], self.array[i]
            i = (i - 1) // 2


    def delete(self, i):
        "delete element at index i"
        if i == -1:
            return False
        self.array[i] = self.array[-1]
        self.array.pop()
        #self.minHeapify(i)
        while True:
            left = self.left(i)
            right = self.right(i)
            smallest = i
            if left<len(self.array) and self.array[left].second < self.array[smallest].second:
                smallest = left
            if right<len(self.array) and self.array[right].second < self.array[smallest].second:
                smallest = right
            if smallest != i:
                self.array[i], self.array[smallest] = self.array[smallest], self.array[i]
                i = smallest
            else:
                break
        return True
    

    def delete_element(self, elt):
        """Delete a specific element from the Min Heap."""
        i = -1
        for j in range(len(self.array)):
            if self.array[j].first == elt:
                i = j
                break
        return self.delete(i)

    
    def minHeapify(self, i):
        """Heapify function to maintain the heap property.""" 
        n = len(self.array)
        smallest = i
        left = self.left(i)
        right = self.right(i)

        if left<n and self.array[left].second < self.array[smallest].second:
            smallest = left
        if right<n and self.array[right].second < self.array[smallest].second:
            smallest = right
        if smallest != i:
            self.array[i], self.array[smallest] = self.array[smallest], self.array[i]
            self.minHeapify(smallest)


    def getMin(self):
        return self.array[0].get() if self.array!=[] else None

    def pop(self):
        if self.array!=[]:
            pair = self.array[0].get()
            self.delete(0)
            return pair
        else:
            return None
    
    def fancy_print(self):
        def aux(i, nb_space):
            if i<len(self.array):
                aux(self.left(i), nb_space+1)
                print(nb_space*"  ", self.array[i])
                aux(self.right(i), nb_space+1)
        aux(0, 0)