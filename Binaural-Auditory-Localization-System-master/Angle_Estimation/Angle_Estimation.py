# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

angle_array=[-90,-60,-30,0,30,60,90];n_angle=range(0,8)

# 讀取資料
word='coffee';angle='0'
filename='IC'+'\IC_'+word+'_'+angle

IC_ch1 = np.loadtxt(filename+'_ch1.txt');IC_ch2 = np.loadtxt(filename+'_ch2.txt')
IC_ch3 = np.loadtxt(filename+'_ch3.txt');IC_ch4 = np.loadtxt(filename+'_ch4.txt')
IC_ch5 = np.loadtxt(filename+'_ch5.txt');IC_ch6 = np.loadtxt(filename+'_ch6.txt')
IC_ch7 = np.loadtxt(filename+'_ch7.txt');IC_ch8 = np.loadtxt(filename+'_ch8.txt')
IC_ch9 = np.loadtxt(filename+'_ch9.txt');IC_ch10 = np.loadtxt(filename+'_ch10.txt')
IC_ch11 = np.loadtxt(filename+'_ch11.txt');IC_ch12 = np.loadtxt(filename+'_ch12.txt')
IC_ch13 = np.loadtxt(filename+'_ch13.txt');IC_ch14 = np.loadtxt(filename+'_ch14.txt')
IC_ch15 = np.loadtxt(filename+'_ch15.txt');IC_ch16 = np.loadtxt(filename+'_ch16.txt')
IC_ch17 = np.loadtxt(filename+'_ch17.txt');IC_ch18 = np.loadtxt(filename+'_ch18.txt')
IC_ch19 = np.loadtxt(filename+'_ch19.txt');IC_ch20 = np.loadtxt(filename+'_ch20.txt')
IC_ch21 = np.loadtxt(filename+'_ch21.txt');IC_ch22 = np.loadtxt(filename+'_ch22.txt')
IC_ch23 = np.loadtxt(filename+'_ch23.txt');IC_ch24 = np.loadtxt(filename+'_ch24.txt')
IC_ch25 = np.loadtxt(filename+'_ch25.txt');IC_ch26 = np.loadtxt(filename+'_ch26.txt')
IC_ch27 = np.loadtxt(filename+'_ch27.txt');IC_ch28 = np.loadtxt(filename+'_ch28.txt')
IC_ch29 = np.loadtxt(filename+'_ch29.txt');IC_ch30 = np.loadtxt(filename+'_ch30.txt')
IC_ch31 = np.loadtxt(filename+'_ch31.txt');IC_ch32 = np.loadtxt(filename+'_ch32.txt')
IC_ch33 = np.loadtxt(filename+'_ch33.txt');IC_ch34 = np.loadtxt(filename+'_ch34.txt')
IC_ch35 = np.loadtxt(filename+'_ch35.txt');IC_ch36 = np.loadtxt(filename+'_ch36.txt')
IC_ch37 = np.loadtxt(filename+'_ch37.txt');IC_ch38 = np.loadtxt(filename+'_ch38.txt')
IC_ch39 = np.loadtxt(filename+'_ch39.txt');IC_ch40 = np.loadtxt(filename+'_ch40.txt')

# 各Channel判斷角度
est_ch1=IC_ch1.argmax(axis=0);est_ch2=IC_ch2.argmax(axis=0)
est_ch3=IC_ch3.argmax(axis=0);est_ch4=IC_ch4.argmax(axis=0)
est_ch5=IC_ch5.argmax(axis=0);est_ch6=IC_ch6.argmax(axis=0)
est_ch7=IC_ch7.argmax(axis=0);est_ch8=IC_ch8.argmax(axis=0)
est_ch9=IC_ch9.argmax(axis=0);est_ch10=IC_ch10.argmax(axis=0)
est_ch11=IC_ch11.argmax(axis=0);est_ch12=IC_ch12.argmax(axis=0)
est_ch13=IC_ch13.argmax(axis=0);est_ch14=IC_ch14.argmax(axis=0)
est_ch15=IC_ch15.argmax(axis=0);est_ch16=IC_ch16.argmax(axis=0)
est_ch17=IC_ch17.argmax(axis=0);est_ch18=IC_ch18.argmax(axis=0)
est_ch19=IC_ch19.argmax(axis=0);est_ch20=IC_ch20.argmax(axis=0)
est_ch21=IC_ch21.argmax(axis=0);est_ch22=IC_ch22.argmax(axis=0)
est_ch23=IC_ch23.argmax(axis=0);est_ch24=IC_ch24.argmax(axis=0)
est_ch25=IC_ch25.argmax(axis=0);est_ch26=IC_ch26.argmax(axis=0)
est_ch27=IC_ch27.argmax(axis=0);est_ch28=IC_ch28.argmax(axis=0)
est_ch29=IC_ch29.argmax(axis=0);est_ch30=IC_ch30.argmax(axis=0)
est_ch31=IC_ch31.argmax(axis=0);est_ch32=IC_ch32.argmax(axis=0)
est_ch33=IC_ch33.argmax(axis=0);est_ch34=IC_ch34.argmax(axis=0)
est_ch35=IC_ch35.argmax(axis=0);est_ch36=IC_ch36.argmax(axis=0)
est_ch37=IC_ch37.argmax(axis=0);est_ch38=IC_ch38.argmax(axis=0)
est_ch39=IC_ch39.argmax(axis=0);est_ch40=IC_ch40.argmax(axis=0)

est_allch_t=np.concatenate([est_ch1,est_ch2,est_ch3,est_ch4,est_ch5,est_ch6,est_ch7,est_ch8,est_ch9,est_ch10,
             est_ch11,est_ch12,est_ch13,est_ch14,est_ch15,est_ch16,est_ch17,est_ch18,est_ch19,est_ch20,
             est_ch21,est_ch22,est_ch23,est_ch24,est_ch25,est_ch26,est_ch27,est_ch28,est_ch29,est_ch30,
             est_ch31,est_ch32,est_ch33,est_ch34,est_ch35,est_ch36,est_ch37,est_ch38,est_ch39,est_ch40])
est_allch_t=np.reshape(est_allch_t,(40,-1))


# 各Channel投票權重
est_weight=np.concatenate(([3], 5*np.ones(21), 3*np.ones(12), np.ones(6)))
est_weight_matrix=np.zeros([40,150])

start=0
for c in range(0,40):
    est_weight_matrix[c,start:start+est_weight[c]]=1
    start=start+est_weight[c]
	
	
# 各時間點的角度估計結果
est_angle=np.zeros((150,len(IC_ch1.T)))
out=np.zeros((7,len(IC_ch1.T)))

for t in range(0,len(IC_ch1.T)):
    est_allch_weight=np.dot(est_allch_t[:,t], est_weight_matrix)
    est_angle[:,t]=est_allch_weight.T
    out[:,t]=np.histogram(est_angle[:,t],n_angle)[0]
    plt.scatter(t*np.ones(7),angle_array,out[:,t]+1,color='r')

plt.yticks(angle_array)
plt.xlabel('Time (ms)');plt.ylabel('Estimated Angle (degree)')
plt.show()