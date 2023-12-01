# 实现AC多模式匹配算法

import time
import sys
from collections import defaultdict

class Node:
    def __init__(self,state_num,ch=None):
        self.state_num = state_num
        self.ch = ch
        self.children = []

class Trie(Node):
    def __init__(self):
        Node.__init__(self,0)
    def init(self):
        # The number that represents each node
        self._state_num_max = 0
        self.goto_dic = defaultdict(lambda :-1)
        self.fail_dic = defaultdict(int)
        self.output_dic = defaultdict(list)
    
    def build(self,patterns):
        for pattern in patterns:
            self.build_all_pattern(pattern)
        self.build_fail()

    def build_all_pattern(self,pattern):
        # 将所有的模式串添加到字典树中
        # 定义一个current变量来表示当前节点
        current = self
        for x in pattern:
            #判断这个模式串中的字符是不是这个当前节点的子节点
            index = self.childnode_in_current(current,x)
            if index == -1:
                #如果返回-1,说明在当前节点没有这个x这个子节点，于是需要新创建一个节点,并且跳转到新创建的节点
                current = self.child_addto_current(current,x)
            else:
                #如果有的话，就直接跳转到那个子节点
                current = current.children[index]
        # 在这个模式串的结尾，将可输出的模式串添加到对应的节点上
        self.output_dic[current.state_num] = [pattern]
                
    def childnode_in_current(self,current,x):
        for i in range(len(current.children)):
            child = current.children[i]
            if child.ch == x:
                # 找到了就返回子节点的索引号
                return i
        # 找不到就返回-1
        return -1
    
    # 为当前节点添加新的子节点并更新current为新的子节点
    def child_addto_current(self,current,x):
        self._state_num_max +=1
        next_node = Node(self._state_num_max,x)
        current.children.append(next_node)
        # 修改转向函数
        self.goto_dic[(current.state_num,x)] = self._state_num_max
        return  next_node
    # fail function
    def build_fail(self):
        node_at_level = self.children
        while node_at_level:
            node_at_next_level = []
            for parent in node_at_level:
                node_at_next_level.extend(parent.children)
                for child in parent.children:
                    v = self.fail_dic[parent.state_num]
                    while self.goto_dic[(v,child.ch)] == -1 and v != 0:
                        v = self.fail_dic[v]
                    fail_value = self.goto_dic[(v,child.ch)]
                    self.fail_dic[child.state_num] = fail_value
                    if self.fail_dic[child.state_num] != 0:
                        self.output_dic[child.state_num].extend(self.output_dic[fail_value])
            node_at_level = node_at_next_level

class AC(Trie):
    def __init__(self):
        Trie.__init__(self)

    def init(self,patterns):
        Trie.init(self)
        self.build(patterns)

    def go_to(self,s,x):
        if s==0:
            if (s,x) not in self.goto_dic:
                return 0
        return self.goto_dic[(s,x)]
    
    def fail(self,s):
        return self.fail_dic[s]
    
    def output(self,s):
        return self.output_dic[s]
    
    def search(self,text):
        current_state = 0
        index = 0
        i = 0
        while index < len(text):
            ch = text[index]
            if self.go_to(current_state,ch)==-1:
                current_state = self.fail(current_state)
            
            current_state = self.go_to(current_state,ch)
            patterns = self.output(current_state)
            if patterns:
                # 每匹配到一个模式串i就+1
                i += 1
                print(current_state,*patterns)
            index += 1
        return i
if __name__ == "__main__":
    # 选择要读入的模式串
    choice = input("input the number of pattern:")
    if choice == '0':
        print(f'程序已退出')
        sys.exit()
    if choice == '1':
        start_time = time.time()
        with open('pattern1w.txt','r') as file:
            patterns = file.readlines()
        patterns = [line.strip() for line in patterns]
    elif choice == '2':
        start_time = time.time()
        with open('pattern2w.txt','r') as file:
            patterns = file.readlines()
        patterns = [line.strip() for line in patterns]
    elif choice == '3':
        start_time = time.time()
        with open('pattern3w.txt','r') as file:
            patterns = file.readlines()
        patterns = [line.strip() for line in patterns]
    # 输入要读入的文本串
    with open('text.txt','r') as text:
        for_text = list(text.read())
    for_text = [line.strip('\n') for line in for_text]
    
    ac = AC()
    ac.init(patterns)
    end_time = time.time()
    # 计算执行时间
    execution_time = end_time - start_time

    # 计算匹配模式串所需要的时间
    start_time = time.time()
    num = ac.search(for_text)
    end_time = time.time()
    execution_time1 = end_time - start_time

    print(f"读入模式串生成字典树和读入文本串所需要的时间：{execution_time}秒")
    print(f"匹配模式串所需要的时间：{execution_time1}秒")
    print(f"有{num}个模式串得到了匹配")

    
            