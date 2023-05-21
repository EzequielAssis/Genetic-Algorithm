import os
import imageio
import matplotlib
import numpy as np
import random as rd
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

class GA(object):
    def __init__(self, ngenes, nchrom, npop, ng, lmin, lmax, tcros, tmut, elt, const_f6, nrm, max_scale):
        self.tcros = tcros
        self.tmut = tmut
        self.npop = npop
        self.lmin = np.array(lmin)
        self.lmax = np.array(lmax)
        self.nchrom = nchrom
        self.ngenes = ngenes
        self.ng = ng
        self.elt = elt
        self.nrm = nrm
        self.const_f6 = const_f6
        self.max_scale = max_scale
        self.bestFit = np.zeros((ng))
        self.avrGen = np.zeros((ng))
        self.worstFit = np.zeros((ng))
        self.pop = np.random.randint(0, 2, size=(self.npop, self.nchrom, self.ngenes))
        self.popInit = self.pop.copy()
    
    def rand_pop(self):
        self.pop = np.random.randint(0, 2, size=(self.npop, self.nchrom, self.ngenes))
        self.popInit = self.pop.copy()

    def bin2int(self, arr, sum=0, index=0):
        if index == len(arr):
            return sum
        else:
            return self.bin2int(arr, sum+arr[-index-1]*(2**index), index+1)

    def int2real(self, arr):
        varList = []
        for val in zip(arr, self.lmin, self.lmax):
            vReal = (val[0]*(val[2]-val[1])/((2**self.ngenes)-1))+val[1]
            varList.append(vReal)
        return varList

    def bin2real(self, arr):
        pop_reshape = arr.reshape((self.nchrom*self.npop, self.ngenes))
        intPop = np.array(list(map(self.bin2int, pop_reshape)))
        intPop = intPop.reshape((self.npop, self.nchrom))
        realPop = np.array(list(map(self.int2real, intPop)))
        return realPop

    def function(self, arr):
        return self.const_f6-(((np.sin(np.sqrt((arr**2).sum()))**2)-0.5)/(1+(0.001*(arr**2).sum()))**2)

    def fitness(self, arr):
        return np.array(list(map(self.function, arr)))

    def norm(self, apt, min):
        if self.nrm:
            indices = np.argsort(apt)
            apt = sorted(apt)
            popSorted = np.zeros(self.pop.shape)
            for c in range(self.npop):
                popSorted[c] = self.pop[indices[c]]
            self.pop = popSorted.copy()
            array = np.zeros(self.npop)
            for i in range(self.npop):
                array[i] = min+((self.max_scale-min)/(len(array)-1))*(i)
            return array
        else:
            return apt

    def selection(self, arr):
        prob = arr/arr.sum()
        #Roleta de seleção
        arrSelection = rd.choices(self.pop.tolist(), prob, k=len(arr))
        return arrSelection

    def crossover(self, arr):
        def rsp(x):
            return np.array(x).reshape(-1)

        fList = []
        cont = 0
        halfPop = int(len(arr)/2)
        popCros = int(halfPop*self.tcros)
        pCortes = (self.nchrom*self.ngenes) - 1

        for _ in range(halfPop):
            par1 = rd.choices(arr)[0]
            par2 = rd.choices(list(filter(lambda x: x != par1, arr)))[0]
            if cont < popCros:
                pc = rd.randint(1, pCortes)
                fList.append(np.append(rsp(par1)[:pc], rsp(par2)[pc:]).reshape((self.nchrom, self.ngenes)).tolist())
                fList.append(np.append(rsp(par2)[:pc], rsp(par1)[pc:]).reshape((self.nchrom, self.ngenes)).tolist())
                cont += 1
            else:
                fList.append(par1)
                fList.append(par2)
        return np.array(fList)

    def mutation(self, arr):
        arrMut = np.random.choice((0,2), size=arr.shape, p=[(1-self.tmut), (self.tmut)])
        arr = arr+arrMut
        arr[arr==2] = 1
        arr[arr==3] = 0
        return arr

    def elitism(self, bestFitCurPop, indexBestFitCurPop, arrNextPop):
        if self.elt:
            arrRealNextPop = self.bin2real(arrNextPop)
            fitNextPop = self.fitness(arrRealNextPop)
            if fitNextPop.max() >= bestFitCurPop:
                return arrNextPop
            else:
                indexWorstNextPop = fitNextPop.argmin()
                arrNextPop[indexWorstNextPop] = self.pop[indexBestFitCurPop]
                return arrNextPop
        else:
            return arrNextPop
    
    def save_matrices(self, ger):
        fig = plt.figure(figsize=(8,8))
        cmap = matplotlib.colors.ListedColormap(['white', 'darkblue'])
        img = plt.imshow(ga.pop.reshape(self.npop, self.ngenes*self.nchrom), cmap=cmap)
        plt.colorbar(img, ticks=[0, 1])
        plt.title(f'geração {ger}')
        plt.xlabel('genes')
        plt.ylabel('indivívuos')
        pathName = os.path.join(os.getcwd(), 'images')
        if not os.path.exists(pathName):
            os.makedirs(pathName)
        filename = os.path.join(pathName, f'{ger}.png')
        plt.savefig(filename)
        plt.close()

    def run(self):
        print('Executando GA...')
        for g in tqdm(range(self.ng)):
            arrReal = self.bin2real(self.pop)
            arrFitness = self.fitness(arrReal)
            arrFitnessNorm = self.norm(arrFitness, 1)
            arrSelection = self.selection(arrFitnessNorm)
            arrNewPop = self.crossover(arrSelection)
            arrMutation = self.mutation(arrNewPop)
            popWithElitism = self.elitism(arrFitness.max(), arrFitness.argmax(), arrMutation)
            self.pop = popWithElitism.copy()
            self.bestFit[g] = arrFitness.max()
            self.avrGen[g] = arrFitness.mean()
            self.worstFit[g] = arrFitness.min()
            self.save_matrices(g)
        print('Execução finalizada!\n')


def generate_gif():
    print('\nGerando GIF...')
    with imageio.get_writer('gif_geracoes.gif', mode='I') as writer:
        for i in range(ga.ng):
            image = imageio.v2.imread(os.path.join(os.getcwd(), 'images', f'{i}.png'))
            writer.append_data(image)
    for i in range(ga.ng-2):
        os.remove(os.path.join(os.getcwd(), 'images', f'{i+1}.png'))
    print('Finalizado!')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ng', type=int, default=100, help='numero de geracoes')
    parser.add_argument('--ngenes', type=int, default=25, help='numero de genes ou bits por cromossomo')
    parser.add_argument('--npop', type=int, default=100, help='numero de individuos')
    parser.add_argument('--nchrom', type=int, default=2, help='numero de cromossomos ou variaveis')
    parser.add_argument('--lmin', type=int, nargs='+', default=(-100, -100), help='minimo das variaveis (cromossomos) no domininio dos reais')
    parser.add_argument('--lmax', type=int, nargs='+', default=(100, 100), help='maximo das variaveis (cromossomos) no domininio dos reais')
    parser.add_argument('--tcros', type=float, default=0.7, help='taxa de cruzamento')
    parser.add_argument('--tmut', type=float, default=0.01, help='taxa de mutacao')
    parser.add_argument('--elt', type=bool, default=False, help='elitismo singular')
    parser.add_argument('--const_f6', type=float, default=0.5, help='constante da f6')
    parser.add_argument('--nrm', type=bool, default=False, help='normalizacao linear')
    parser.add_argument('--max_scale', type=float, default=100, help='escala maxima da normalizacao (se --norm True)')
    return parser.parse_args()


args = parse_opt()
ga = GA(**vars(args))
ga.run()
generate_gif()
