from subprocess import Popen, PIPE
import signal
import os
import time
f = open('state.txt','r')
pid = int(f.read())
f.close()
f =open('state.txt','w')
if pid==0:
    print("Starting Carla")
    process = Popen(['~/Downloads/CARLA_0.9.5/CarlaUE4.sh', '-ResX=8','ResY=8'], stdout=PIPE, stderr=PIPE,shell=True)
    f.write(str(process.pid))
else:
    print("Kill Carla")

    os.kill(pid,signal.SIGTERM)
    time.sleep(2)
    print("Starting Carla")
    process = Popen(['~/Downloads/CARLA_0.9.5/CarlaUE4.sh', '-ResX=8','ResY=8'], stdout=PIPE, stderr=PIPE,shell=True)
    f.write(process.pid)

f.close()

# print(process)