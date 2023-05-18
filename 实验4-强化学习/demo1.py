##GridWorld
# 引入所需要的库文件
import numpy as np
from plot_utils_TD import plot_values, plot_values_change
import xml.dom.minidom


# 在内存中创建一个空的文档
doc = xml.dom.minidom.Document()
# 创建一个根节点Managers对象
root = doc.createElement('Begin')
doc.appendChild(root)

#定义世界的一些基本参数，世界的长和宽需要用户手动输入：
world_h = int(input("Enter a world_h: "))  # 5
world_w = int(input("Enter a world_w: "))  # 6
length = world_h * world_w
gamma = 1
state = [i for i in range(length)]
action = ['n', 'e', 's', 'w']
ds_action = {'n': -world_w, 'e': 1, 's': world_w, 'w': -1}
value = [0 for i in range(length)]


# 定义回报函数
def reward(s):
    return 0 if s in [0, length - 1] else -1


# 动态回归需要的备忘录
def next_states(s, a):
    next_state = s
    if (s < world_w and a == 'n') or (s % world_w == 0 and a == 'w') \
            or (s > length - world_w - 1 and a == 's') or (s % (world_w - 1) == 0 and a == 'e' and s != 0):
        pass
    else:
        next_state = s + ds_action[a]
    return next_state


# 过程函数
def getsuccessor(s):
    successor = []
    for a in action:
        next = next_states(s, a)
        nodeName = doc.createElement('next')
        # 给叶子节点name设置一个文本节点，用于显示文本内容
        nodeName.appendChild(doc.createTextNode(str(next)))
        root.appendChild(nodeName)

        successor.append(next)
    return successor
    # 状态更新函数


def value_update(s):
    value_new = 0
    if s in [0, length - 1]:
        pass
    else:
        successor = getsuccessor(s)
        rewards = reward(s)
        for next_state in successor:
            value_new += 1.00 / len(action) * (rewards + gamma * value[next_state])
            nodevalue = doc.createElement('value_new')
            # 给叶子节点name设置一个文本节点，用于显示文本内容
            nodevalue.appendChild(doc.createTextNode(str(value_new)))
            nodeReward = doc.createElement('reward')
            nodeReward.appendChild(doc.createTextNode(str(rewards)))
            root.appendChild(nodevalue)
            root.appendChild(nodeReward)
    return value_new


# 调用上述方法
def main():
    max_iter = 10
    global value
    v = np.array(value).reshape(world_h, world_w)
    print(v)
    iter = 0
    while iter < max_iter:
        new_value = [0 for i in range(length)]
        for s in state:
            new_value[s] = value_update(s)
        value = new_value
        v = np.array(value).reshape(world_h, world_w)
        print(v)
        plot_values_change(value, world_h, world_w)
        iter += 1
    # 开始写xml文档
    fp = open('E:\大三学期\大三下学期\机器学习\实验\机器学习-实验\实验4-强化学习\GridWorld.xml', 'w')


doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding="utf-8")

