"""
Created on Wed Feb 24 21:24:03 2021
@author: mockingbird
"""

import pygame
from itertools import product
import collections
from time import sleep, perf_counter
from random import sample
from pprint import pprint

poids1 = [[36, 12, 12], [4, 1, 1],[4, 1, 0]]
poids2 = [[8, 7, 6], [5, 4, 3], [2, 1, 0]]
poids3 = poids2
poids4 = [[8, 7, 6], [5, 3, 2], [4, 1, 0]]
poids5 = poids4
poids6 = [[1, 1, 1], [1, 1, 1],[ 1, 1, 0]]

div1 = div3 = div5 = 4
div2 = div4 = div6 = 1

data = [[1,8, 2], [0, 4, 3], [7, 6, 5]]
# data = [[2,3,6], [1, 4, 7], [5, 8, 0]]
etat_final = [[1,2,3],[4,5,6],[7,8,0]]

# data = [[15,2,1,12],[8,5,6,11],[4,9,10,7],[3,14,13,0]]
# etat_final = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]]

size = (600, 800)[len(data) == 4]

"""
################################################################################
############                                                        ############
############                        Classe Node                     ############
############                                                        ############
################################################################################
"""
class Node:

    def __init__(self, puzzle, parent=None, action=None):
        self.puzzle = puzzle
        self.parent = parent
        self.action = action
        
        if (self.parent != None):
            self.g = parent.g + 1
        else:
            self.g = 0
            
    @property
    def state(self):
        return str(self)

    @property 
    def path(self): 
        node, p = self, []
        while node:
            p.append(node)
            node = node.parent
        yield from reversed(p)

    @property
    def solved(self):
        return self.puzzle.solved # bool

    @property
    def actions(self):
        return self.puzzle.actions

    @property
    def heuristique(self):
        # return self.puzzle.mal_place
        return self.puzzle.manhattan_ponderer
        
    @property
    def function_eval(self): 
        return self.heuristique + self.g

    def __str__(self):
        return str(self.puzzle)

"""
################################################################################
############                                                        ############
############                      Classe Solver                     ############
############                                                        ############
################################################################################
"""
class Solver:

    def __init__(self, start):
        self.start = start

    def solve_Astar(self):
        queue = collections.deque([Node(self.start)])
        seen = set()
        seen.add(queue[0].state)
        while queue:
            queue = collections.deque(sorted(list(queue), key = lambda node: node.function_eval))
            node = queue.popleft()
            
            if node.solved:
                return node.path, len(seen), len(queue)

            for move, action in node.actions:
                child = Node(move(), node, action)
                if child.state not in seen:
                    queue.append(child)
                    seen.add(child.state)
        
    def solve_profondeur(self):
        queue = collections.deque([Node(self.start)])
        seen = set()
        print(queue[0].state)
        seen.add(queue[0].state)
        while queue:
            node = queue.pop()
            
            if node.solved:
                return node.path, len(seen), len(queue)

            for move, action in node.actions:
                child = Node(move(), node, action)             
                if child.state not in seen:
                    queue.append(child)
                    seen.add(child.state)

    def solve_largeur(self):
        queue = collections.deque([Node(self.start)])
        seen = set()
        seen.add(queue[0].state)
        while queue:
            node = queue.popleft()
            
            if node.solved:
                return node.path, len(seen), len(queue)

            for move, action in node.actions:
                child = Node(move(), node, action)             
                if child.state not in seen:
                    queue.append(child)
                    seen.add(child.state)  

"""
################################################################################
############                                                        ############
############                      Classe Puzzle                     ############
############                                                        ############
################################################################################
"""
class Puzzle:

    def __init__(self, data, etat_final=etat_final):
        self.data = data
        self.etat_final = etat_final
        
    @property
    def solved(self):
        return str(self) == ''.join(map(str, range(1, pow(len(self.data), 2)))) + '0'

    @property 
    def actions(self):
        def create_move(at, to):
            return lambda: self._move(at, to)

        moves = []
        for i, j in product(range(len(self.data)), range(len(self.data))):
            
            directions = {'droite': (i, j-1),'gauche': (i, j+1), 'bas': (i-1, j),'haut': (i+1, j)}
            
            for action, (r, c) in directions.items():
                
                if r >= 0 and c >= 0 and r < len(self.data) and c < len(self.data) and self.data[r][c] == 0:
                    
                    move = create_move((i,j), (r,c)), action
                    moves.append(move)
        return moves

    @property
    def mal_place (self):
        piecesMalPlace = 0
        for i in range(len(self.data)):
            for j in range(len(self.data)):
                if self.data[i][j] != self.etat_final[i][j]:
                    piecesMalPlace += 1
        return piecesMalPlace   # valeur
    
    def getPosition_x_y(self, board, num):
        for i in range(len(self.data)):
            for j in range(len(self.data)):
                if board[i][j] == num:
                    return i, j
    
    # poids pour les peices de la 1er ligne 
    @property
    def manhattan_ponderer(self):
        final = 0;
        for x in range(len(self.data)):
            for y in range(len(self.data)):
                i , j = self.getPosition_x_y(self.data, self.data[x][y])
                i_f, j_f = self.getPosition_x_y(self.etat_final, self.data[x][y])
                
                final += ((abs(i - i_f) + abs(j - j_f)) * poids1[i_f][j_f])
                
        return final // div1
    
    def copy(self):
        data = []
        for row in self.data:
            data.append([x for x in row])
        return Puzzle(data)

    def _move(self, at, to):
        copy = self.copy()
        i, j = at
        r, c = to
        copy.data[i][j], copy.data[r][c] = copy.data[r][c], copy.data[i][j]
        return copy

    def pprint(self):
        for row in self.data:
            print(row)
        print()

    def __str__(self):
        return ''.join(map(str, self))

    def __iter__(self):
       for row in self.data:
            yield from row
            
"""
################################################################################
############                                                        ############
############                      Classe Images                     ############
############                                                        ############
################################################################################
"""     
class Images :
    
    def __init__(self, data):
        self.ecran = pygame.display.set_mode((size, size))
        pygame.display.set_caption('Le Taquin __Projet - IA__')
        pygame.display.set_icon(pygame.image.load("image/taquin.png"))
        self.data = data
        
    def loadImages(self):
        
        "afficher les images sur le canvas"
        # 1er palier
        position1 = pygame.image.load("image/"+str(self.data[0][0])+".png")
        self.ecran.blit(position1, (24, 24))
        position2 = pygame.image.load("image/"+str(self.data[0][1])+".png")
        self.ecran.blit(position2, (225, 24))
        position3 = pygame.image.load("image/"+str(self.data[0][2])+".png")
        self.ecran.blit(position3, (426, 24))
        
        if(len(self.data) == 4):
             position4 = pygame.image.load("image/"+str(self.data[0][3])+".png")
             self.ecran.blit(position4, (626, 24))

        # 2nd palier
        position5 = pygame.image.load("image/"+str(self.data[1][0])+".png")
        self.ecran.blit(position5, (24, 225))
        position6 = pygame.image.load("image/"+str(self.data[1][1])+".png")
        self.ecran.blit(position6, (225, 225))
        position7 = pygame.image.load("image/"+str(self.data[1][2])+".png")
        self.ecran.blit(position7, (426, 225))
        
        if(len(self.data) == 4):
             position8 = pygame.image.load("image/"+str(self.data[1][3])+".png")
             self.ecran.blit(position8, (626, 224))
        
        # 3 palier
        position9 = pygame.image.load("image/"+str(self.data[2][0])+".png")
        self.ecran.blit(position9, (24, 426))
        position10 = pygame.image.load("image/"+str(self.data[2][1])+".png")
        self.ecran.blit(position10, (225, 426))
        position11 = pygame.image.load("image/"+str(self.data[2][2])+".png")
        self.ecran.blit(position11, (426, 426))
        
        if(len(self.data) == 4):
            position12 = pygame.image.load("image/"+str(self.data[2][3])+".png")
            self.ecran.blit(position12, (626, 424))

            # 4 palier
            position13 = pygame.image.load("image/"+str(self.data[3][0])+".png")
            self.ecran.blit(position13, (24, 626))
            position14 = pygame.image.load("image/"+str(self.data[3][1])+".png")
            self.ecran.blit(position14, (225, 626))
            position15 = pygame.image.load("image/"+str(self.data[3][2])+".png")
            self.ecran.blit(position15, (426, 626))
            position16 = pygame.image.load("image/"+str(self.data[3][3])+".png")
            self.ecran.blit(position16, (626, 626))
            
    def fill(self, color):
        self.ecran.fill(color)
        
    def update(self, new_value, time=0.3):
        self.data = new_value
        self.loadImages()
        pygame.display.flip()
        sleep(time)
        
"""
################################################################################
############                                                        ############
############                    Classe Listener                     ############
############                                                        ############
################################################################################
"""  
class Melangeur:
    
    def __init__(self, data):
        self.data = data
    
    @property
    def isSolvable(self) :
        inv_count = 0
        for i in range(0, len(self.data)-1) :
            for j in range(i + 1, len(self.data)) : 
                if (self.data[j][i] > 0 and self.data[j][i] > self.data[i][j]) :
                    inv_count += 1
        return (inv_count % 2 == 0)
    
    def permutation(self, source, source1, destination, destination1):
        tmp = self.data[source][source1]
        self.data[source][source1] = self.data[destination][destination1]
        self.data[destination][destination1] = tmp
        
    @property
    def melange(self):
        for i in range(10):
            l = sample(range(len(self.data)), len(self.data)-1)
            l1 = sample(range(len(self.data)), len(self.data)-1)
            self.permutation(l[0], l1[0], l[1], l1[1])
        return self.data
    
    @property
    def getSolvable(self):
        while self.isSolvable :
            self.melange

"""
################################################################################
############                                                        ############
############                        Classe Jeu                      ############
############                                                        ############
################################################################################
""" 
class Jeu :
    
    global solver
    
    def __init__(self, data=data):
        self.data = data
        self.play = True
        self.exec = Melangeur(self.data)
        self.images = Images(self.data)
        
    def Astar(self):
        solver = Solver(Puzzle(self.data))
        begin = perf_counter()
        p, v, e = solver.solve_Astar()
        end = perf_counter()
        compteur = 0
        if p != None:
            for node in p:
                print("coup {}:".format(compteur))
                node.puzzle.pprint()
                self.images.update(node.puzzle.data, 0.1)
                # self.data = node.puzzle.data
                compteur += 1
            print("Solution en : " + str(compteur-1)+ " coups")
            print("Total des noeuds vue: " + str(v))
            print("Total des noeuds expense: " + str(e))
            print("Temps : " + str(end - begin) + " seconde(s)")
        else:
            pprint("Erreur inconnue, veuillez redemarer !")
        
    def profondeur(self):
        solver = Solver(Puzzle(self.data))
        begin = perf_counter()
        p, v, e = solver.solve_profondeur()
        end = perf_counter()
        compteur = 0
        if p != None:
            for node in p:
                print("coup {}:".format(compteur))
                node.puzzle.pprint()
                self.images.update(node.puzzle.data, 0.1)
                compteur += 1
            print("Solution en : " + str(compteur-1)+ " coups")
            print("Total des noeuds vue: " + str(v))
            print("Total des noeuds expense: " + str(e))
            print("Temps : " + str(end - begin) + " seconde(s)")
        else:
            pprint("Erreur inconnue, veuillez redemarer !")
    
    def longeur(self):
        solver = Solver(Puzzle(self.data))
        begin = perf_counter()
        p, v, e = solver.solve_largeur()
        end = perf_counter()
        compteur = 0
        if p != None:
            for node in p:
                print("coup {}:".format(compteur))
                node.puzzle.pprint()
                self.images.update(node.puzzle.data, 0.1)
                # self.data = node.puzzle.data
                compteur += 1
            print("Solution en : " + str(compteur-1)+ " coups")
            print("Total des noeuds vue: " + str(v))
            print("Total des noeuds expense: " + str(e))
            print("Temps : " + str(end - begin) + " seconde(s)")
        else:
            pprint("Erreur inconnue, veuillez redemarer !")

    def main(self):
        while self.play:
            for even in pygame.event.get():
                if even.type == pygame.QUIT:
                    self.play = False
                    
                if even.type == pygame.KEYDOWN:
                    if even.key == pygame.K_c:
                        self.data = self.exec.melange
                        
                        if(not self.exec.isSolvable): # si non solvable
                            self.exec.getSolvable
                            
                        self.images.update(self.data, 0)
                        pygame.display.flip()
                        pprint(self.exec.data)
                        
                    if even.key == pygame.K_SPACE:
                        pprint("-------résolution avec A* ---------------")
                        self.Astar()
                        
                    if even.key == pygame.K_p:
                        pprint("------- résolution avec recherche en profondeur ---------------")
                        self.profondeur()                        
                         
                    if even.key == pygame.K_l:
                        pprint("------- résolution avec recherche en largeur ---------------")
                        self.longeur()
                        
                    if even.key == pygame.K_q:
                        self.start = False
        
            self.images.fill((0, 0, 0))
            self.images.loadImages()
            pygame.display.flip()

if __name__ == '__main__':
    pygame.init()
    Jeu().main()
    pygame.quit()