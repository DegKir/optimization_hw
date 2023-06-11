import numpy as np
import cvxpy as cp
import time
import scipy.stats as st
class GradientDescent:
    def __init__(self, f, gradient_fn,learning_rate,
                 start_theta,new_point_formula='classical_gradient', step_choose='constant_step',stopping_criteria='delta_theta', max_iterations=1000, delta_theta=0.0000001, delta_f=0.0000001, delta_grad=0.0000001, additional_data=[]):
        self.f = f                               #Функция
        self.gradient_fn = gradient_fn           #Градиент
        self.learning_rate=learning_rate         #Константа обучения
        self.theta = start_theta                 #Начальная точка
        self.max_iterations = max_iterations     #Максимальное число итераций
        self.delta_theta = delta_theta           #Точность по координатам
        self.delta_f = delta_f                   #Точность по функционалу
        self.delta_grad = delta_grad             #Точность по градиенту

        self.additional_data=additional_data     #При необходимости передавать сюда матрицу A и вектор b
        self.some_numbers=[] #Произвольные числа, прим. из критериев
        self.fstory=[self.f(self.theta)]                           #Храним историю значений функций
        self.story=[start_theta]                 #Храним историю точек

        #Выбор конкретного метода спуска
        if new_point_formula =='classical_gradient':
            self.next_point=self.classical_gradient
        elif new_point_formula == 'mirror':
            self.next_point=self.mirror
        elif new_point_formula == 'wolf':
            self.next_point=self.Frank_Wolf
        elif new_point_formula == 'sim_SGD':
            self.next_point=self.sim_SGD
        elif new_point_formula == 'sim_SGD_batch':
            self.next_point=self.sim_SGD_batch
        elif new_point_formula == 'SGD':
            self.next_point=self.SGD
        elif new_point_formula == 'SAGA':
            point_size=len(self.theta)
            batch_amount=len(self.additional_data[1])
            self.additional_data.append(np.zeros((batch_amount,point_size))) # Создаем массив для хранения запоздалых точек градиента
            for point in self.additional_data[-1] :                             # Инициализируем стартовыми точками
                point=self.theta
            self.next_point=self.SAGA
        elif new_point_formula == 'SVRG':
            point_size=len(self.theta)
            self.additional_data.append(0) #Число итераций внутреннего цикла
            self.additional_data.append(np.zeros(point_size)) #Хранится опорная точка фи
            additional_data[-1]=self.theta
            self.next_point=self.SVRG
        else:
            raise Value_error('Invalid method for new point calculation')

        #Выбор конкретного способа делать шаг
        if step_choose   == 'constant_step':
            self.learning_rate_fn=self.constant_step
        elif step_choose == 'declining_step':
            self.learning_rate_fn=self.declining_step
        elif step_choose == 'argmin_step':
            self.learning_rate_fn=self.argmin_step
        elif step_choose == 'armijo_rule':
            self.learning_rate_fn=self.armijo_rule
        elif step_choose == 'wolf_rule':
            self.learning_rate_fn=self.wolf_rule
        elif step_choose == 'goldstein_condition':
            self.learning_rate_fn=self.goldstein_condition
        elif step_choose == 'pol_shor':
            self.learning_rate_fn=self.pol_shor
        elif step_choose == 'optimal':
            self.learning_rate_fn=self.optimal
        else:
            raise ValueError('Invalid step_choose')

        #Выбор конкретного критерия остановки
        if stopping_criteria == 'delta_theta':
            self.stopping_fn = self.delta_theta_stopping
        elif stopping_criteria == 'delta_f':
            self.stopping_fn = self.delta_f_stopping
        elif stopping_criteria == 'delta_grad':
            self.stopping_fn = self.delta_grad_stopping
        elif stopping_criteria == 'gap':
            self.stopping_fn = self.gap_stopping
        elif stopping_criteria == 'exact_sol':
            self.stopping_fn = self.exact_stopping
        else:
            raise ValueError('Invalid stopping criteria')

    #Методы градиентного спуска
    ###Может сделать шаг внутренним полем ?
    def classical_gradient(self,step):
        return self.theta-step*self.gradient_fn(self.theta)

    #Зеркальный
    def mirror(self,step):
        answer=np.empty((len(self.theta)))
        grad=self.gradient_fn(self.theta)

        #Считаем знаменатель
        summ=0
        for i in range(len(self.theta)):
            summ+=self.theta[i]*np.exp(-step*grad[i])

        #Считаем числители и делим
        for i in range(len(self.theta)):
            answer[i]=self.theta[i]*np.exp(-step*grad[i])/summ
        return answer


    def sim_SGD(self,step):
        norm = st.norm(0,100)
        return self.theta-step*(self.gradient_fn(self.theta)+np.array(norm.rvs(len(self.theta))))

        #Размер батча передается в additional_data
    def sim_SGD_batch(self,step):
        batch_size=self.additional_data
        summ = np.zeros(len(self.theta))
        norm = st.norm(0,100)
        for j in range(batch_size):
            summ+=np.array(norm.rvs(len(self.theta)))
        summ*=1/batch_size
        return self.theta-step*(self.gradient_fn(self.theta)+summ)

    #В additional_data передаем первым аргументом массив функций, вторым массив градиентов
    def SGD(self,step):
        n=len(self.additional_data[1])      #Количество батчей
        uni=st.randint(0,n)                 #Равномерный выбор градиента
        gradients=self.additional_data[1]
        gradient=gradients[uni.rvs()]
        return self.theta-step*gradient(self.theta)

    def SAGA(self,step):
        n=len(self.additional_data[1])      #Количество батчей
        gradients=self.additional_data[1]
        uni=st.randint(0,n)                 #Равномерный выбор градиента
        j=uni.rvs()
        #Считаем сумму заранее
        summ=0
        for i in range(0,n):
            summ+=gradients[i](self.additional_data[-1][i])
        #Заменяем один из градиентов новым
        old_theta=self.additional_data[-1][j]
        self.additional_data[-1][j]=self.theta
        gk=gradients[j](self.theta)-gradients[j](old_theta)+summ/n
        return self.theta-step*gk

    def SVRG(self,step):
        n=len(self.additional_data[1])      #Количество батчей
        gradients=self.additional_data[1]
        phi=self.additional_data[-1]
        m=self.additional_data[-2]          #Число итераций внутреннего цикла на данный момент
        uni=st.randint(0,n)                 #Равномерный выбор градиента
        j=uni.rvs()

        gk=gradients[j](self.theta)-gradients[j](phi)+self.gradient_fn(phi)
        if(m>=n):#Если внутренний цикл подошел к концу то нужно взять среднее последних m точек
            summ=0
            xes=self.story[-m:] # Беру последние m точек
            for point in xes:
                summ+=point     #Прохожусь по ним и складываю
            self.additional_data[-1]=summ/m
            self.additional_data[-2]=0
        return self.theta-step*gk
    #минимизация линейной функции на вероятностном симплексе, ответ - x_I=1, I- наименьший индекс у вектора задающего функционал
    def Frank_Wolf(self,step):
        grad=self.gradient_fn(self.theta)
        min_index = np.argmin(grad)
        answer=np.zeros(len(self.theta))
        answer[min_index]=1;
        return self.theta+step*(answer-self.theta)

    #Критерии остановки
    def delta_theta_stopping(self):
        self.some_numbers.append(np.linalg.norm(self.theta-self.story[-1]))
        return np.linalg.norm(self.theta - self.story[-1]) < self.delta_theta

    def delta_f_stopping(self):
        f_curr = self.f(self.theta)
        self.some_numbers.append(np.abs(f_curr - fstory[-1]))
        return np.abs(f_curr - fstory[-1]) < self.delta_f

    def delta_grad_stopping(self):
        grad_norm = np.linalg.norm(self.gradient_fn(self.theta))
        self.some_numbers.append(grad_norm**2)
        return grad_norm**2 < self.delta_grad

    def exact_stopping(self):
        x_exact=self.additional_data[2]
        self.some_numbers.append(np.linalg.norm(self.theta-x_exact))
        return np.linalg.norm(self.theta-x_exact) < self.delta_theta

    #Достаем из широких штанин солвер для симплекс метода
    def gap_stopping(self):
        grad=self.gradient_fn(self.theta)
        min_index = np.argmin(grad)
        answer=np.zeros(len(self.theta))
        answer[min_index]=1;
        self.some_numbers.append(np.matmul(grad,self.theta-answer))
        return np.matmul(grad,self.theta-answer) < self.delta_f

    #Виды шага

    def constant_step(self):
        step=1/self.learning_rate
        return step

    #1/(k+1) шаг
    def declining_step(self):
        step=1/(len(self.theta)+self.learning_rate)
        return step

    #Аргминимум
    def argmin_step(self):
        x=self.theta
        A=self.additional_data[0]
        g=self.gradient_fn(x)
        verh=np.dot(g.T,g)
        niz=np.dot(np.dot(g.T,A),g)
        step=verh/niz
        return step

    #Правило Армихо
    def armijo_rule(self):
        x=self.theta
        A=self.additional_data[0]
        b=self.additional_data[1]
        c=self.additional_data[3]
        g=self.gradient_fn(x)
        niz=np.dot(np.dot(g.T,A),g)
        verh=-c*np.dot(g.T,g)-np.dot(b.T,g)+np.dot(np.dot(x.T,A),g)
        step=2*verh/niz
        return step
    #Правило волка
    def wolf_rule(self):
        x=self.theta
        A=self.additional_data[0]
        b=self.additional_data[1]
        c1=self.additional_data[3]
        c2=self.additional_data[4]
        g=self.gradient_fn(x)
        niz=np.dot(np.dot(g.T,A),g)
        left=-c2*np.dot(g.T,g)+np.dot(np.dot(g.T,A),x)+np.dot(g.T,b)
        right=2*(-c1*np.dot(g.T,g)-np.dot(b.T,g)+np.dot(np.dot(x.T,A),g))
        left=left/niz
        right=right/niz
        step=left+(right-left)/2
        return step

    def goldstein_condition(self):
        x=self.theta
        A=self.additional_data[0]
        b=self.additional_data[1]
        c=self.additional_data[3]
        g=self.gradient_fn(x)
        niz=np.dot(np.dot(g.T,A),g)
        left=2*(-(1-c)*np.dot(g.T,g)+np.dot(np.dot(g.T,A),x)-np.dot(g.T,b))
        right=2*(-c*np.dot(g.T,g)+np.dot(np.dot(x.T,A),g)-np.dot(b.T,g))
        left=left/niz
        right=right/niz
        step=left+(right-left)/2
        return step

    def pol_shor(self):
        x=self.theta
        x_exact=self.additional_data[2]
        g=self.gradient_fn(x)
        alpha=self.additional_data[3]
        verh=self.f(x)-self.f(x_exact)
        niz=alpha*np.dot(g.T,g)
        step=verh/niz
        return step

    def optimal(self):
        mu=self.additional_data[0]
        L=self.learning_rate
        step=1/(mu+L)
        return step

    def compute(self):
        i=0
        times=[]
        iteration_start = time.time()
        for i in range(self.max_iterations):
            grad = self.gradient_fn(self.theta)
            step = self.learning_rate_fn()
            self.story.append(self.theta)
            self.fstory.append(self.f(self.theta))
            self.theta=self.next_point(step)
            times.append(time.time()-iteration_start)
            if self.stopping_fn():
                break
        print("amount of iterations is {}".format(i))
        return self.theta, times,self.some_numbers
