``` 
export do="--dry-run=client -o yaml"
export now="--force --grace-period 0"
vim ~/.vimrc
set tabstop=2
set expandtab
set shiftwidth=2
alias kn='kubectl config set-context --current --namespace '
usage: kn mynamespace

k expose pod <podname> --name <svc-name> --port 5000 --targetPort 80





kubectl get -l app=design2 pod
kubectl edit pod design2-766d48574f-5w274

 kubectl exec -c simpleapp -it try1-5db9bc6f85-whxbf \
-- /bin/bash -c 'echo $ilike'
kubectl exec -it try1-d4fbf76fd-46pkb -- /bin/bash -c 'env'

View a file within the new volume mounted in a container. It should match the data we created inside the configMap.
Because the file did not have a carriage-return it will appear prior to the following prompt.
student@cp:˜$ kubectl exec -c simpleapp -it try1-7865dcb948-stb2n \
-- /bin/bash -c 'cat /etc/cars/car.trim'


helm -n mercury ls -a

apiVersion: v1
kind: Pod
metadata:
  name: pod6
  namespace: default
spec:
  containers:
  - name: shra
    image: busybox:1.31.0
    readinessProbe:
      initialDelaySeconds: 5
      periodSeconds: 10
      exec:
        command:
        - cat
        - /tmp/ready
    args:
    - sh
    - -c
    - touch /tmp/ready && sleep 1d
	
	
	
apiVersion: v1
kind: Pod
metadata:
  name: nginx
  labels:
    app.kubernetes.io/name: proxy
spec:
  containers:
  - name: nginx
    image: nginx:stable
    ports:
      - containerPort: 80
        name: http-web-svc
        
---
apiVersion: v1
kind: Service
metadata:
  name: nginx-service
spec:
  selector:
    app.kubernetes.io/name: proxy
  ports:
  - name: name-of-service-port
    protocol: TCP
    port: 80
    targetPort: http-web-svc
	
	
k rollout history deployment/api-new-c32 -n neptune
k rollout history deployment/api-new-c32 -n neptune --revision=5 > 5.out
k rollout undo deployment/api-new-c32 --to-revision=3 -n neptune --dry-run
k describe deploy api-new-c32 -n neptune

k run shra --image=nginx:alpine -n pluto
k run xxx --image --dry-run=client -o yaml > xxx.yaml
kubectl run nginx --image=nginx --restart=Never --dry-run=client -n mynamespace -o yaml > pod.yaml
k run mypod1 -n mynamespace --image=nginx $do
kubectl create deployment design2 --image=nginx
kubectl create deployment newserver --image=httpd
commandhelp:    etcdctl snapshot restore -h
k delete pod --force --grace-period 0
test deployment is successful
curl $(kubectl get svc my-app-svc -o jsonpath="{.spec.clusterIP}")
version-1


JOBS
kubectl create job pi  --image=perl -- /bin/sh -c perl -Mbignum=bpi -wle 'print bpi(2000)'
kubectl get jobs -w

containers
kubectl logs secondapp webserver
			 <podname> <containername>
kubectl exec  -it  secondapp -c busy -- sh

endpoint for service
kubectl get ep secondapp -o yaml	

ps -elf |grep kube-proxy
vim /var/log/kube-proxy.log
journalctl -a | grep proxy		
student@cp: ̃$ kubectl -n kube-system get pod
student@cp: ̃$ kubectl -n kube-system logs kube-proxy-fsdfr
Check that the proxy is creating the expected rules for the problem service. Find the destination port being used for theservice,32000in this case.i
sudo iptables-save |grep secondapp

sudo docker build -t registry.killer.sh:5000/sun-cipher/v1-docker:latest .
sudo docker images | grep sun-cipher
sudo docker push registry.killer.sh:5000/sun-cipher/v1-docker:latest
sudo docker images rm <>
sudo podman run --name sun-cipher -d registry.killer.sh:5000/sun-cipher:v1-podman
-d background position sensitive
build a docker image and export it as a tgz
docker save myimage:latest | gzip > myimage_latest.tar.gz

secrets
echo "mega_secret_key" | base64
echo "bWVnYV9zZWNyZXRfa2V5Cg==" | base64 -d



 helm list
   42  helm list -n mercury
   43  helm delete -h
   44  helm uninstall internal-issue-report-apiv1
   45  helm uninstall internal-issue-report-apiv1 -n mercury
   46  helm list -n mercury
   47  helm update -h
   48  helm upgrade -h
   49  helm chart list
   50  helm chart -h
   51  helm list -n mercury
   52  helm upgrade -h
   53  helm upgrade internal-issue-report-apiv2 bitnami/nginx -n mercury
   54  helm list -n mercury
   55  helm install -h
   56  helm install internal-issue-report-apache bitnami/apache -n mercury --dry-run
   57  helm install internal-issue-report-apache bitnami/apache -n mercury --dry-run -o yaml
   58  helm install -h
   59  helm list -n mercury
   60  helm install internal-issue-report-apache bitnami/apache --set replicas=2 -n mercury --dry-run -o yaml
   61  helm install internal-issue-report-apache bitnami/apache --set replicas=2 -n mercury --dry-run
   62  helm install internal-issue-report-apache bitnami/apache --set replicaCount=2 -n mercury --dry-run
   63  helm install internal-issue-report-apache bitnami/apache --set replicaCount=2 -n mercury
   65  helm list -n mercury
   66  helm describe -h
   75  helm list --all-namespaces
   76  helm -n mercury ls -a
   77  helm uninstall internal-issue-report-daniel
   78  helm uninstall internal-issue-report-daniel -n mercury

sudo podman logs sun-cipher
sudo podman ps        //running containers

labels & annotations
k get po -n sun -l type=worker
k label po -n sun -l type=runner protected=true
k get po -n sun --show-labels
kubectl -n sun annotate pods -l protected=true protected="do not delete this pod"

deployment with 3 pods and 1 container
apiVersion: apps/v1
kind: Deployment
metadata:
  annotations:
    deployment.kubernetes.io/revision: "1"
  generation: 1
  labels:
    run: stressmeout
  name: neptune-10ab
  namespace: neptune
spec:
  replicas: 3
  selector:
    matchLabels:
      run: stressmeout
  strategy:
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 1
    type: RollingUpdate
  template:
    metadata:
      creationTimestamp: null
      labels:
        run: stressmeout
    spec:
      serviceAccountName: neptune-sa-v2
      containers:
      - image: httpd:2.4-alpine
        imagePullPolicy: Always
        name: neptune-pod-10ab
        resources:
          limits:
            memory: "50Mi"
	      requests:
            memory: "20Mi"
        terminationMessagePath: /dev/termination-log
        terminationMessagePolicy: File
      dnsPolicy: ClusterFirst
      restartPolicy: Always
      schedulerName: default-scheduler
      securityContext: {}
      terminationGracePeriodSeconds: 30
	  
deployment with version
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app-v1
  labels:
    app: my-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-app
      version: v1
  template:
    metadata:
      labels:
        app: my-app
        version: v1
    spec:
      containers:
      - name: nginx
        image: nginx
        ports:
        - containerPort: 80
        volumeMounts:
        - name: workdir
          mountPath: /usr/share/nginx/html
      initContainers:
      - name: install
        image: busybox:1.28
        command:
        - /bin/sh
        - -c
        - "echo version-1 > /work-dir/index.html"
        volumeMounts:
        - name: workdir
          mountPath: "/work-dir"
      volumes:
      - name: workdir
        emptyDir: {}
		
while sleep 0.1; do curl $(kubectl get svc my-app-svc -o jsonpath="{.spec.clusterIP}"); done


job with command
apiVersion: batch/v1
kind: Job
metadata:
  creationTimestamp: null
  name: job1
spec:
  template:
    metadata:
      creationTimestamp: null
    spec:
      containers:
      - image: busybox
        name: job1
        resources: {}
        args:
        - sh
        - -c
        - 'echo hello;sleep 30;echo world'
      restartPolicy: Never
status: {}


POD with env var via configmap
apiVersion: v1
kind: Pod
metadata:
  creationTimestamp: null
  labels:
    run: nginx
  name: nginx
spec:
  containers:
  - image: nginx
    imagePullPolicy: IfNotPresent
    name: nginx
    resources: {}
    env:
    - name: option # name of the env variable
      valueFrom:
        configMapKeyRef:
          name: options # name of config map
          key: var5 # name of the entity in config map
  dnsPolicy: ClusterFirst
  restartPolicy: Never
status: {}
```


```

kubectl create quota myrq --hard=cpu=1,memory=1G,pods=2 --dry-run=client -o yaml
kubectl run nginx --image=nginx --restart=Never --port=80
kubectl set image pod/nginx nginx=nginx:1.7.1
kubectl get po nginx -o jsonpath='{.spec.containers[].image}{"\n"}'
kubectl run busybox --image=busybox --rm -it --restart=Never -- wget -O- 10.1.1.131:80
kubectl logs nginx --previous
kubectl run busybox --image=busybox -it --restart=Never -- echo 'hello world'
# or
kubectl run busybox --image=busybox -it --restart=Never -- /bin/sh -c 'echo hello world'
k create deployment nginx --image=nginx:1.18.0 -r 2 --port=80 $do > deploynginx.yaml
kubectl set image deploy nginx nginx=nginx:1.19.8
k autoscale deployment.apps/nginx --min=5 --max=10 --cpu-percent=80
k scale deployment.apps/nginx --replicas=5
k rollout pause deployment.apps/nginx

k label pod nginx2 --overwrite app=v2
k get pod -L app
k label pods tier=web -l='app in (v1, v2)'
remove label app from pods containing label app
kubectl label po -l app app-
kubectl label nodes <your-node-name> accelerator=nvidia-tesla-p100
spec->nodeSelector property
k explain po.spec
kubectl annotate pod nginx1 --list
k annotate po nginx{1..3} description-


kubectl -n qa get events | grep -i "Liveness probe failed" | awk '{print $4}'
kubectl get events | grep -i error

kubectl top nodes




   1  k get namesapce
    2  k get ns
    3  k get ns > /opt/course/1/namespaces
    4  ls
    5  vim mypod.yaml
    6  touch /opt/course/2/pod1-status-command.sh
    7  k create -f mypod.yaml
    8  k get pod pod1 -o yaml
    9  k get pod pod1 -o json
   10  k get pod pod1 -o json | jq status.phase
   11  k get pod pod1 -o json | jq '.status.phase'
   12  vim /opt/course/2/pod1-status-command.sh
   13  vim /opt/course/3/job.yaml
   14  k create -f /opt/course/3/job.yaml
   15  k get jobs
   16  k get jobs -n neptune
   17  k describe job neb-new-job | tail -n 15
   18  k describe job neb-new-job -n neptune | tail -n 15
   19  vim /opt/course/3/job.yaml
   20  k delete job neb-new-job -n neptune
   21  k create -f /opt/course/3/job.yaml
   22  k get jobs -n neptune
   23  k describe job neb-new-job -n neptune | tail -n 15
   24  k get jobs -n neptune
   25  k describe job neb-new-job -n neptune
   26  /bin/sleep 2 && echo "done"
   27  k get pod
   28  k get pod -n neptune
   29  k logs neb-new-job-pmjfl
   30  k logs neb-new-job-pmjfl -n neptune
   31  vim /opt/course/3/job.yaml
   32  k delete job neb-new-job -n neptune
   33  k create -f /opt/course/3/job.yaml
   34  k get pod -n neptune
      35  k get jobs -n neptune
   36  k logs neb-new-job
   37  k logs neb-new-job -n neptune
   38  k describe job neb-new-job -n neptune
   39  k set namespace mercury
   40  k set -h
   41  helm list
   42  helm list -n mercury
   43  helm delete -h
   44  helm uninstall internal-issue-report-apiv1
   45  helm uninstall internal-issue-report-apiv1 -n mercury
   46  helm list -n mercury
   47  helm update -h
   48  helm upgrade -h
   49  helm chart list
   50  helm chart -h
   51  helm list -n mercury
   52  helm upgrade -h
   53  helm upgrade internal-issue-report-apiv2 bitnami/nginx -n mercury
   54  helm list -n mercury
   55  helm install -h
   56  helm install internal-issue-report-apache bitnami/apache -n mercury --dry-run
   57  helm install internal-issue-report-apache bitnami/apache -n mercury --dry-run -o yaml
   58  helm install -h
   59  helm list -n mercury
   60  helm install internal-issue-report-apache bitnami/apache --set replicas=2 -n mercury --dry-run -o yaml
   61  helm install internal-issue-report-apache bitnami/apache --set replicas=2 -n mercury --dry-run
   62  helm install internal-issue-report-apache bitnami/apache --set replicaCount=2 -n mercury --dry-run
   63  helm install internal-issue-report-apache bitnami/apache --set replicaCount=2 -n mercury
   64  k get pods -n mercury
   65  helm list -n mercury
   66  helm describe -h
      67  k describe pod internal-issue-report-apache-69c84588-fx4gf | tail -n 15
   68  k describe pod internal-issue-report-apache-69c84588-fx4gf -n mercury | tail -n 15
   69  k describe pod internal-issue-report-apache-69c84588-fx4gf -n mercury
   70  k logs internal-issue-report-apache-69c84588-fx4gf -n mercury
   71  k describe pod internal-issue-report-apache-69c84588-fx4gf -n mercury -o json | grep pending
   72  k get pod internal-issue-report-apache-69c84588-fx4gf -n mercury -o json | grep pending
   73  k describe internal-issue-report-apache-69c84588-fx4gf -n mercury -o json | grep pending
   74  k get pod internal-issue-report-apache-69c84588-fx4gf -n mercury -o json
   75  helm list --all-namespaces
   76  helm -n mercury ls -a
   77  helm uninstall internal-issue-report-daniel
   78  helm uninstall internal-issue-report-daniel -n mercury
   79  vim mypod6.yaml
   80  k create -f mypod6.yaml
   81  vim mypod6.yaml
   82  k create -f mypod6.yaml
   83  k get pod
   84  k describe pod pod6
   85  ls /tmp/
   86  vim mypod6.yaml
   87  k delete pod pod6
   88  k get po
   89  k create -f mypod6.yaml
   90  k get po
   91  k describe pod pod6
   92  k get po
   93  k delete pod pod6
   94  k get po
   95  vim mypod6.yaml
   96  k get po
   97  k create -f mypod6.yaml
   98  k get po
     99  vim mypod6.yaml
  100  k create -f mypod6.yaml
  101  k describe pod pod6
  102  k describe pod pod6 | tail -n 15
  103  vim mypod6.yaml
  104  k delete pod pod6
  105  k create -f mypod6.yaml
  106  k describe pod pod6 | tail -n 15
  107  k get po
  108  k exec pod6 -it -- ls /tmp
  109  k describe pod pod6 | tail -n 15
  110  vim mypod6.yaml
  111  k get pods -n saturn
  112  k describe pods webserver-sat-001 -n saturn | grep my-happy-shop
  113  k describe pods webserver-sat-002 -n saturn | grep my-happy-shop
  114  k describe pods webserver-sat-003 -n saturn | grep my-happy-shop
  115  k get pod webserver-sat-003 -n saturn -o yaml > my-happy.yaml
  116  vim my-happy.yaml
  117  k create -f my-happy.yaml
  118  k delete pod webserver-sat-003 -n saturn
  119  k get deploy
  120  k get deploy -n neptune
  121  k describe deploy api-new-c32 -n neptune
  122  k rollout history deployment/api-new-c32 -n neptune
  123  k rollout history deployment/api-new-c32 -n neptune --revision=5
  124  k rollout history deployment/api-new-c32 -n neptune --revision=4
  125  k describe deploy api-new-c32 -n neptune
  126  k rollout history deployment/api-new-c32 -n neptune
  127  k logs api-new-c32 -n neptune
  128  k rollout undo
  129  k rollout undo deployment/api-new-c32 -n neptune
  130  k rollout history deployment/api-new-c32 -n neptune
    131  k describe deploy api-new-c32 -n neptune
  132  vim /opt/course/9/holy-api-deployment.yaml
  133  vim /opt/course/9/holy-api-pod.yaml
  134  vim /opt/course/9/holy-api-deployment.yaml
  135  vim /opt/course/9/holy-api-pod.yaml
  136  vim /opt/course/9/holy-api-deployment.yaml
  137  k create -f /opt/course/9/holy-api-deployment.yaml
  138  vim /opt/course/9/holy-api-deployment.yaml
  139  k create -f /opt/course/9/holy-api-deployment.yaml
  140  vim /opt/course/9/holy-api-deployment.yaml
  141  k create -f /opt/course/9/holy-api-deployment.yaml
  142  vim /opt/course/9/holy-api-deployment.yaml
  143  k create -f /opt/course/9/holy-api-deployment.yaml
  144  vim /opt/course/9/holy-api-deployment.yaml
  145  k create -f /opt/course/9/holy-api-deployment.yaml
  146  vim myservice10.yaml
  147  k apply -f myservice10.yaml
  148  vim myservice10.yaml
  149  k get pods
  150  k delete pod project-plt-6cc-api
  151  k delete -f myservice10.yaml
  152  vim myservice10.yaml
  153  k apply -f myservice10.yaml
  154  k get svc,po -n pluto
  155  k get svc,po -n pluto -o wide
  156  k run -it --image=nginx:alpine -n pluto -- /bin/bash
  157  k exec -it --image=nginx:alpine -n pluto -- /bin/bash
  158  k run -it --image=nginx:alpine -n pluto
  159  k run shra -it --image=nginx:alpine -n pluto
  160  k run shra --image=nginx:alpine -n pluto
  161  k delete pod shra -n pluto
  162  k run shra --image=nginx:alpine -n pluto
  163  k exec shra -it -- /bin/bash
  164  k exec shra -n pluto -it -- /bin/bash
    165  k exec shra -n pluto -it -- /bin/sh
  166  vim myservice10.yaml
  167  k apply -f myservice10.yaml
  168  k exec shra -n pluto -it -- /bin/sh
  169  vim /opt/course/10/service_test.html
  170  k get pods -n pluto
  171  k logs project-plt-6cc-api -n pluto
  172  vim /opt/course/10/service_test.log
  173  k delete pod shra -n pluto
  174  ls
  175  ls /opt/course/11
  176  ls /opt/course/11/image/
  177  vim /opt/course/11/image/Dockerfile
  178  pwd
  179  cd ~
  180  echo ~
  181  cd /opt/course/11/image/
  182  docker build -t registry.killer.sh:5000/sun-cipher/v1-docker:latest .
  183  sudo docker build -t registry.killer.sh:5000/sun-cipher/v1-docker:latest .
  184  docker images | grep sun-cipher
  185  sudo docker images | grep sun-cipher
  186  sudo podman build -t registry.killer.sh:5000/sun-cipher/v1-podman .
  187  sudo docker images | grep sun-cipher
  188  sudo podman images | grep sun-cipher
  189  sudo docker push registry.killer.sh:5000/sun-cipher/v1-docker:latest
  190  sudo podman push registry.killer.sh:5000/sun-cipher/v1-podman
  191  sudo podman run -h
  192  sudo podman run --help
  193  sudo podman run --name sun-cipher -d registry.killer.sh:5000/sun-cipher:v1-podman
  194  sudo podman logs --help
  195  sudo podman logs sun-cipher
  196  vim /opt/course/11/logs
    197  sudo podman ls
  198  sudo podman ps
  199  history | grep helm
  200  k get deploy -n moon
  201  k get deploy -n moon -o yaml > deploy15.yaml
  202  vim deploy15.yaml
  203  vim mycm.yaml
  204  k create cm --help
  205  k create cm --help --from-file=/opt/course/15/web-moon.html -n moon
  206  k create cm --from-file=/opt/course/15/web-moon.html -n moon
  207  k create cm configmap-web-moon-html --from-file=/opt/course/15/web-moon.html -n moon
  208  k get cm configmap-web-moon-html -o yaml
  209  k get cm configmap-web-moon-html -n moon -o yaml
  210  k edit cm configmap-web-moon-html -n moon -o yaml
  211  k create pod temp --image=nginx:alpine
  212  k run temp --image=nginx:alpine
  213  k get po
  214  k delete po temp
  215  k run temp --image=nginx:alpine -n moon
  216  k get po
  217  k get po -n moon
  218  k exec temp -it -- /bin/sh
  219  k exec temp -it -n moon -- /bin/sh
  220  k des po -n moon
  221  k describe po -n moon
  222  k get cm -n moon
  223  k describe cm configmap-web-moon-html
  224  k describe cm configmap-web-moon-html -n moon
  225  k get deploy web-moon
  226  k get deploy web-moon -n moon
  227  k describe deploy web-moon -n moon
  228  k get pod -n moon  197  sudo podman ls
  198  sudo podman ps
  199  history | grep helm
  200  k get deploy -n moon
  201  k get deploy -n moon -o yaml > deploy15.yaml
  202  vim deploy15.yaml
  203  vim mycm.yaml
  204  k create cm --help
  205  k create cm --help --from-file=/opt/course/15/web-moon.html -n moon
  206  k create cm --from-file=/opt/course/15/web-moon.html -n moon
  207  k create cm configmap-web-moon-html --from-file=/opt/course/15/web-moon.html -n moon
  208  k get cm configmap-web-moon-html -o yaml
  209  k get cm configmap-web-moon-html -n moon -o yaml
  210  k edit cm configmap-web-moon-html -n moon -o yaml
  211  k create pod temp --image=nginx:alpine
  212  k run temp --image=nginx:alpine
  213  k get po
  214  k delete po temp
  215  k run temp --image=nginx:alpine -n moon
  216  k get po
  217  k get po -n moon
  218  k exec temp -it -- /bin/sh
  219  k exec temp -it -n moon -- /bin/sh
  220  k des po -n moon
  221  k describe po -n moon
  222  k get cm -n moon
  223  k describe cm configmap-web-moon-html
  224  k describe cm configmap-web-moon-html -n moon
  225  k get deploy web-moon
  226  k get deploy web-moon -n moon
  227  k describe deploy web-moon -n moon
  228  k get pod -n moon
    229  k describe po web-moon-66c79744fb-56lvh -n moon
  230  k exec temp -it -n moon -- /bin/sh
  231  k get pod -n moon -o wide
  232  k get deploy web-moon -n moon
  233  k get deploy web-moon -n moon -o wide
  234  k get svc -n moon -o wide
  235  k exec temp -it -n moon -- /bin/sh
  236  k get deploy -n mercury
  237  k get deploy -n mercury -o yaml
  238  ls /opt/course/16/
  239  cp /opt/course/16/cleaner.yaml /opt/course/16/cleaner-new.yaml
  240  vim /opt/course/16/cleaner-new.yaml
  241  k get pod -n mercury
  242  k apply -f /opt/course/16/cleaner-new.yaml
  243  vim /opt/course/16/cleaner-new.yaml
  244  k apply -f /opt/course/16/cleaner-new.yaml
  245  k get deploy -n mercury -o yaml
  246  k get deploy -n mercury
  247  k get pod -n mercury
  248  k describe pod cleaner-64cd7dfcf5-pvl8n -n mercury
  249  vim /opt/course/16/cleaner-new.yaml
  250  k get pod -n mercury
  251  k logs cleaner-64cd7dfcf5-pvl8n logger-con -n mercury
  252  bash
  253  vim /opt/course/11/logs
  254  vim /opt/course/16/cleaner-new.yaml
  255  k apply -f myservice10.yaml
  256  k apply -f /opt/course/16/cleaner-new.yaml
  257  k get pods -n pluto
  258  k get deploy,pods -n mercury
  259  k logs pod/cleaner-64586c5764-5z5ft logger-con -n mercury
  260  k logs pod/cleaner-64586c5764-2s7zc logger-con -n mercury
    261  vim /opt/course/17/test-init-container.yaml
  262  k create -f /opt/course/17/test-init-container.yaml
  263  k get deploy
  264  k get deploy -n mars
  265  vim /opt/course/17/test-init-container.yaml
  266  k describe deploy test-init-container -n mars
  267  k get deploy,po -n mars
  268  k run temp --image=nginx:alpine -n mars
  269  k exec temp -n mars -it -- /bin/sh
  270  k get deploy,po -n mars
  271  k describe pod pod/test-init-container-58cd7c79d9-fkh75 -n mars
  272  k describe pod test-init-container-58cd7c79d9-fkh75 -n mars
  273  k logs test-init-container-58cd7c79d9-fkh75 -n mars
  274  vim /opt/course/17/test-init-container.yaml
  275  k apply -f /opt/course/17/test-init-container.yaml
  276  k get deploy,po -n mars
  277  k describe pod/test-init-container-bbc67c8ff-qr78b -n mars
  278  k logs pod/test-init-container-bbc67c8ff-qr78b -n mars
  279  vim /opt/course/17/test-init-container.yaml
  280  k delete /opt/course/17/test-init-container.yaml
  281  k delete -f /opt/course/17/test-init-container.yaml
  282  k create -f /opt/course/17/test-init-container.yaml
  283  k get deploy,po -n mars
  284  k describe pod/test-init-container-655bd94969-vhvzq -n mars
  285  k logs test-init-container-655bd94969-vhvzq -n mars
  286  vim /opt/course/17/test-init-container.yaml
  287  k delete -f /opt/course/17/test-init-container.yaml
  288  k create -f /opt/course/17/test-init-container.yaml
  289  k get deploy,po -n mars
  290  vim deploy21.yaml
  291  k create -f deploy21.yaml
  292  k get depoy,pod -n neptune
    293  k get deploy,pod -n neptune
  294  k delete -f deploy21.yaml
  295  vim deploy21.yaml
  296  k create -f deploy21.yaml
  297  vim deploy21.yaml
  298  k create -f deploy21.yaml
  299  k get deploy,pod -n neptune
  300  k delete -f deploy21.yaml
  301  vim deploy21.yaml
  302  k create -f deploy21.yaml
  303  k get deploy,pod -n neptune
  304  k get sa -n neptune
  305  k delete -f deploy21.yaml
  306  vim deploy21.yaml
  307  k create -f deploy21.yaml
  308  vim deploy21.yaml
  309  k create -f deploy21.yaml
  310  k get deploy,pod -n neptune
  311  vim deploy21.yaml
  312  k get po -n sun
  313  k get po -n sun -l type:worker
  314  k get po -n sun -l type=worker
  315  k get po -n sun --show-labels
  316  k set --help
  317  k set selector --help
  318  k label --help
  319  k label po -n sun -l type=worker protected=treu
  320  k label po -n sun -l type=worker protected=true
  321  k label po -n sun -l type=worker --overwrite protected=true
  322  k get po -n sun --show-labels
  323  k label po -n sun -l type=runner protected=true
  324  k get po -n sun --show-labels
    325  k edit pod --help
  326  k add --h
  327  k add --help
  328  k annotate --help
  329  kubectl annotate pods -l protected=true protected=do not delete this pod -n sun
  330  kubectl annotate pods -l protected=true -n sun protected=do not delete this pod
  331  kubectl annotate pods -n sun -l protected=true protected=do not delete this pod
  332  kubectl -n sun annotate pods -l protected=true protected=do not delete this pod
  333  kubectl -n sun annotate pods -l protected=true protected="do not delete this pod"
  334  sudo docker images | grep sun-cipher
  335  sudo docker push registry.killer.sh:5000/sun-cipher/v1-docker:latest
  336  sudo podman push registry.killer.sh:5000/sun-cipher/v1-podman:latest
  337  ps -ef
  338  ps -ef | grep podman
  339  ps -ef | grep sun-cipher
  340  ps -ef | grep registry-killer
  341  k get deploy,po -n venus
  342  k get deploy,po -n jupiter
  343  k get deploy,po,svc -n jupiter
  344  k edit service/jupiter-crew-svc -n jupiter
  345  k get deploy,po,svc -n jupiter
  346  k get nodes
  347  k get nodes -o wide
  348  curl http://192.168.100.11:30100
  349  curl http://192.168.100.12:30100
  350  curl http://192.168.100.12:30735
  351  curl http://192.168.100.11:30735
  352  k get svc -n mars
  353  k get pods -n mars
  354  k run temp --image=nginx:alpine -n mars
  355  k exec temp -n mars -it -- /bin/sh
  356  k edit svc manager-api-svc -n mars
    357  k get pods -n mars --show-labels
  358  k edit svc manager-api-svc -n mars
  359  k get all -n mars
  360  k exec temp -n mars -it -- /bin/sh
  361  k create secret
  362  k create secret --help
  363  k get all -n moon
  364  vim secret14.yaml
  365  k edit pod/secret-handler -n moon
  366  k create -f secret14.yaml
  367  vim secret14.yaml
  368  k create -f secret14.yaml
  369  k create secret
  370  k create secret user=test -n moon
  371  k create secret secret1 user=test -n moon
  372  vim /opt/course/14/secret2.yaml
  373  k create -f /opt/course/14/secret2.yaml
  374  vim /opt/course/14/secret2.yaml
  375  k create -f /opt/course/14/secret2.yaml
  376  vim secret14.yaml
  377  vim /opt/course/14/secret2.yaml
  378  k create -f secret14.yaml
  379  vim /opt/course/14/secret2.yaml
  380  k create -f secret14.yaml
  381  vim secret14.yaml
  382  echo "test" | base64
  383  vim secret14.yaml
  384  echo "pwd" | base64
  385  vim secret14.yaml
  386  k create -f secret14.yaml
  387  vim /opt/course/14/secret-handler-new.yaml
  388  cp /opt/course/14/secret-handler.yaml /opt/course/14/secret-handler-new.yaml
    389  vim /opt/course/14/secret-handler-new.yaml
  390  k get all -n moon
  391  vim edit pod/secret-handler -n moon
  392  k edit pod/secret-handler -n moon
  393  k get secrets -n moon
  394  k edit pod/secret-handler -n moon
  395  vim /opt/course/14/secret-handler-new.yaml
  396  k apply -f /opt/course/14/secret-handler-new.yaml
  397  k get all -n moon
  398  k delete pod/secret-handler -n moon
  399  k get all -n moon
  400  k create -f /opt/course/14/secret-handler-new.yaml
  401  k get all -n moon
  402  k describe pod/secret-handler -n moon
  403  k logs pod/secret-handler -n moon
  404  k get pod/secret-handler -n moon -o yaml
  405  k exec -it pod/secret-handler -n moon -- /bin/sh
  406  vim /opt/course/14/secret2.yaml
  407  echo "MTIzNDU2Nzg=" | base64 -d
  408  k create service -h
  409  k create service nodeport -h
  410  k create secret -h
  411  k create secret ass --dry-run -o yaml
  412  k create secret ass -o yaml
  413  k create secret ass
  414  k create secret
  415  k create secret generic -h
  416  k create secret generic shrasecret --from-literal=key1=value1
  417  k get secret/shrasecret
  418  k get secret/shrasecret -o wid
  419  k get secret/shrasecret -o yaml
  420  k create deploy -h
    421  k create deploy shradeploy --image=nginx
  422  k create job -h
  423  k create cronjob -h
  424  k get all -n mars
  425  vim /opt/course/17/test-init-container.yaml
  426  k describe pod/test-init-container-74c4d89b88-8v8tn -n mars
  427  k log pod/test-init-container-74c4d89b88-8v8tn -n mars
  428  k logs pod/test-init-container-74c4d89b88-8v8tn -n mars
  429  vim /opt/course/17/test-init-container.yaml
  430  k apply -f /opt/course/17/test-init-container.yaml
  431  k get all -n mars
  432  k log -n mars pod/test-init-container-7bd5b84478-97hll
  433  k logs -n mars pod/test-init-container-7bd5b84478-97hll
  434  vim /opt/course/17/test-init-container.yaml
  435  k apply -f /opt/course/17/test-init-container.yaml
  436  k get all -n mars
  437  k get secrets
  438  k get secrets shrasecret
  439  k get secrets shrasecret -o yaml
  440  history | grep docker
  441  clear
  442  history
  443  export do="--dry-run=client -o yaml"
  444  export now="--force --grace-period 0"
  445  k create ns -h
  446  k create ns mynamespace
  447  k create pod mypod1 --image=nginx -n mynamespace
  448  k create pod -h
  449  k run pod mypod1 --image=nginx -n mynamespace
  450  k get pod mypod -n mynamespace
  451  k get pod mypod1 -n mynamespace
  452  k get pod -n mynamespace
    453  k get pod -n mynamespace -o yaml
  454  k get pod -n mynamespace
  455  k delete pod pod -n mynamespace
  456  k run mypod1 --image=nginx -n mynamespace
  457  k get -n mynamespace pod mypod1
  458  k get -n mynamespace pod mypod1 -o yaml
  459  k get pods -A
  460  k get pods
  461  k get pods -n mynamespace
  462  k delete pod -n namespace mypod1
  463  k delete pod -n mynamespace mypod1
  464  k run mypod1 -n mynamespace $do
  465  k run mypod1 -n mynamespace --image=nginx $do
  466  k get pods -n mynamespace
  467  k run mypod1 -n mynamespace --image=nginx $do > mypod1.yaml
  468  vim mypod1.yaml
  469  k run --image=busybox -n mynamespace $do
  470  k run busyboxpod --image=busybox -n mynamespace $do
  471  echo $do
  472  export do="--dry-run=client -o yaml"
  473  k delete pod -n mynamespace busyboxpod
  474  k delete pod -n mynamespace busyboxpod $now
  475  export now="--force --grace-period 0"
  476  echo $now
  477  k run busyboxpod --image=busybox -n mynamespace $do
  478  k run busyboxpod --image=busybox -n mynamespace $do > busyboxpod.yaml
  479  vim busyboxpod.yaml
  480  k create -f busyboxpod.yaml
  481  k get all -n mynamespace
  482  k logs pod/busyboxpod -n mynamespace
  483  k describe pod/busyboxpod -n mynamespace
  484  q
    485  k get all -n mynamespace
  486  k describe pod/busyboxpod -n mynamespace
  487  k get all -n mynamespace
  488  vim busyboxpod.yaml
  489  k delete -f busyboxpod.yaml
  490  k create -f busyboxpod.yaml
  491  vim busyboxpod.yaml
  492  k create -f busyboxpod.yaml
  493  vim busyboxpod.yaml
  494  k create -f busyboxpod.yaml
  495  vim busyboxpod.yaml
  496  k create -f busyboxpod.yaml
  497  k run busyboxpod --image=busybox --command env -n mynamespace $do > busyboxpod.yaml
  498  vim busyboxpod.yaml
  499  k create -f busyboxpod.yaml
  500  k get all -n mynamespace
  501  k describe pod/busyboxpod -n mynamespace
  502  k logs pod/busyboxpod -n mynamespace
  503  vim busyboxpod.yaml
  504  k create ns -h
  505  k create ns newns $do
  506  k create resourcequota -h
  507  k create quota myrq --hard=cpu=1,memory=1G,replicas=2 $do
  508  k create quota myrq --hard=cpu=1,memory=1G,pods=2 $do
  509  k run nginx --image=nginx $do
  510  k create pod -h
  511  k run nginx --image=nginx $do > nginx.yaml
  512  vim nginx.yaml
  513  k create -f nginx.yaml
  514  k get pods | grep nginx
  515  vim nginx.yaml
  516  k apply -f nginx.yaml
    517  vim nginx.yaml
  518  k apply -f nginx.yaml
  519  k get pods | grep nginx
  520  k describe pods nginx -n mynamespace
  521  k describe pods nginx
  522  k get pods | grep nginx
  523  k get pod nginx -o wide
  524  k run temp --imae=busybox
  525  k run temp --image=busybox
  526  k exec temp -it -- /bin/sh
  527  k get pod temp
  528  k get pod temp -o yaml
  529  k get pods
  530  k delete pod shradeploy-78d77f88cc-hqprq
  531  k get pods
  532  k get pod temp
  533  k get pods
  534  k delete pod temp
  535  k exec nginx -it -- /bin/sh
  536  k run busybox --image=busybox $do
  537  vim busyboxpod.yaml
  538  k create -f  busyboxpod.yaml
  539  vim busyboxpod.yaml
  540  k delete pod -n mynamespace busyboxpod
  541  k create -f  busyboxpod.yaml
  542  k get podd -n mynamespace
  543  k get pod -n mynamespace
  544  k logs pod/busyboxpod -n mynamespace
  545  k run busybox --image=busybox -- echo "Hello" $do
  546  k get pod -n mynamespace
  547  k logs pod/busyboxpod -n mynamespace
  548  k describe pods -n mynamespace busyboxpod
    549  k run busybox --image=busybox -- echo 'Hello' $do
  550  k delete pod -n mynamespace busyboxpod
  551  k run busybox --image=busybox -- echo 'Hello' $do
  552  k delete pod -n mynamespace busyboxpod $now
  553  k run busybox --image=busybox -- echo 'Hello' $do
  554  k delete pod busybox
  555  k run busybox --image=busybox -- echo 'Hello' $do
  556  k get pod busybox
  557  k describe pod busybox
  558  k logs pod busybox
  559  k logs busybox
  560  k get pod busybox
  561  vim busyboxpod.yaml
  562  k delete pod busybox -n mynamespace $now
  563  k create -f  busyboxpod.yaml
  564  k get pod -n mynamespace
  565  k describe pod busybox -n mynamespace | tail -n 5
  566  vim busyboxpod.yaml
  567  k delete pod busybox -n mynamespace $now
  568  k create -f  busyboxpod.yaml
  569  vim busyboxpod.yaml
  570  k delete pod busybox -n mynamespace $now
  571  k create -f  busyboxpod.yaml
  572  k get pod -n mynamespace
  573  k describe pod busybox -n mynamespace | tail -n 5
  574  k delete pod busybox -n mynamespace $now
  575  k create -f  busyboxpod.yaml
  576  vim busyboxpod.yaml
  577  k delete pod busyboxpod -n mynamespace $now
  578  k create -f  busyboxpod.yaml
  579  k get pod -n mynamespace
  580  k describe pod busybox -n mynamespace | tail -n 5
    581  k describe pod busyboxpod -n mynamespace | tail -n 5
  582  k get pod -n mynamespace
  583  k logs busyboxpod -n mynamespace
  584  vim busyboxpod.yaml
  585  ls
  586  vim mypod1.yaml
  587  vim mypod.yaml
  588  k delete pod busyboxpod -n mynamespace $now
  589  k run busyboxpod --image=busybox $do > busyboxpod.yaml
  590  vim busyboxpod.yaml
  591  k create -f busyboxpod.yaml
  592  vim busyboxpod.yaml
  593  k create -f busyboxpod.yaml
  594  k logs busyboxpod -n mynamespace
  595  k get pod -n mynamespace
  596  vim busyboxpod.yaml
  597  k logs busyboxpod
  598  k get pods
  599  vim busyboxpod.yaml
  600  kubectl run busybox --image=busybox -it --rm --restart=Never -- /bin/sh -c 'echo hello world' $do
  601  kubectl run busybox --image=busybox -it --rm --restart=Never $do
  602  kubectl run busybox --image=busybox --rm --restart=Never $do
  603  kubectl run busybox --image=busybox --restart=Never $do
  604  k run envpod --image=nginx -help
  605  k run envpod --image=nginx -h
  606  k run envpod --image=nginx --env="var1=val1" $do
  607  k run envpod --image=nginx --env="var1=val1"
  608  k exec envpod -it -- /bin/sh
  609  k run busybox2 --image=busybox -h
  610  k run busybox2 --image=busybox --command "echo hello; sleep 3600"
  611  k delete pod busybox2
  612  k run busybox2 --image=busybox --command "echo hello; sleep 3600" $do
    613  k run busybox2 --image=busybox --command "echo hello; sleep 3600" $do > busybox2.yaml
  614  vim busybox2.yaml
  615  k create -f busybox2.yaml
  616  k get pod
  617  k describe pod busybox2
  618  vim busybox2.yaml
  619  k delete pod busybox2
  620  k create -f busybox2.yaml
  621  vim busybox2.yaml
  622  k delete pod busybox2
  623  k create -f busybox2.yaml
  624  k delete pod busybox2
  625  vim busybox2.yaml
  626  k create -f busybox2.yaml
  627  k get pod
  628  k get pod busybox2
  629  k descr pod busybox2
  630  k describe pod busybox2
  631  vim busybox2.yaml
  632  k get pod busybox2
  633  k logs busybox2
  634  k logs busybox2 -c busybox2
  635  k logs busybox2 -c busybox22
  636  vim busybox2.yaml
  637  k get pod busybox2
  638  vim busybox2.yaml
  639  k delete pod busybox2
  640  k delete pod busybox2 $now
  641  k create -f busybox2.yaml
  642  k get pod busybox2
  643  vim busybox2.yaml
  644  k delete pod busybox2 $now
    645  k create -f busybox2.yaml
  646  k get pod busybox2
  647  k logs busybox2 -c busybox22
  648  k logs busybox2 -c busybox2
  649  k exec busybox2 -c busybox22 -it -- ls
  650  vim busybox2.yaml
  651  k exec busybox2 -c busybox22 -it -- /bin/sh
  652  k get pod busybox2
  653  vim busybox2.yaml
  654  k describe pod busybox2
  655  k describe pod busybox2 | tail -n 10
  656  k get pod busybox2
  657  vim busybox2.yaml
  658  k delete pod busybox2 $now
  659  k create -f busybox2.yaml
  660  k get pod busybox2
  661  k describe pod busybox2 | tail -n 10
  662  k get pod busybox2
  663  k describe pod busybox2 | tail -n 10
  664  k get pod busybox2
  665  vim busybox2.yaml
  666  k delete pod busybox2 $now
  667  k create -f busybox2.yaml
  668  k get pod busybox2
  669  k describe pod busybox2 | tail -n 10
  670  k get pod busybox2
  671  k describe pod busybox2 | tail -n 10
  672  k get pod busybox2
  673  k describe pod busybox2 | tail -n 10
  674  k exec busybox2 -c busybox22 -it -- /bin/sh
  675  k describe pod busybox2 | tail -n 10
  676  vim busybox2.yaml
    677  k delete pod busybox2 $now
  678  k create -f busybox2.yaml
  679  k get pod busybox2
  680  k describe pod busybox2 | tail -n 10
  681  vim busybox2.yaml
  682  k describe pod busybox2 | tail -n 10
  683  k get pod busybox2
  684  k get pods
  685  k delete pod nginx
  686  k run nginx --image=nginx -h
  687  k run nginx --image=nginx --port=80 $do > nginx.yaml
  688  vim nginx.yaml
  689  k create -f nginx.yaml
  690  vim nginx.yaml
  691  k create -f nginx.yaml
  692  k get pod
  693  k get pod nginx
  694  k describe pod nginx
  695  k get pod nginx -o wide
  696  k run temp --image=busybox --rm
  697  k run temp --image=busybox
  698  k exec temp -it -- /bin/sh
  699  k get pod temp
  700  k describe pod temp
  701  k get pods
  702  k delete pod envpod
  703  k delete pod envpod $now
  704  k delete pod pod1 $now
  705  k delete pod pod6 $now
  706  k delete pod shradeploy-78d77f88cc-p8shr $now
  707  k get pods
  708  k exec temp -it -- /bin/sh
    709  k run temp --image=temp -h
  710  k run nginx1 --image=temp -l=app=v1
  711  k get pod nginx1 -o yaml
  712  k run nginx2 --image=temp -l=app=v1
  713  k run nginx3 --image=temp -l=app=v1
  714  k get pods --show-labels
  715  k set label -h
  716  k label -h
  717  k label pod nginx2 --overwrite app=v2
  718  k get pods --show-labels
  719  k labels -h
  720  k label -h
  721  k get pods -h
  722  k get -o custom-columns pod -l app
  723  k get pods -o custom-columns -l app
  724  k get pods -l app
  725  k get pods -o custom-columns=.metadata.label -l app
  726  k get pod nginx1 -o yaml
  727  k get pods -o custom-columns=LABELS:.metadata.labels -l app
  728  k get pods -o custom-columns=LABELS:.metadata.labels.app -l app
  729  k get pods -o custom-columns=LABELS:.metadata.labels.app
  730  k get pod -L app
  731  k get pods -l=app=v2
  732  k label -h
  733  k label tier=web -l=app in ('v1' 'v2')
  734  k label tier=web -l=app in ('v1' 'v2');
  735  k label tier=web -l=app in ('v1', 'v2');
  736  k label tier=web -l='app in (v1, v2)'
  737  k label pods tier=web -l='app in (v1, v2)'
  738  k label pods tier=web -l 'app in (v1, v2)'
  739  k annotate -h
  740  k annotate pods owner=marketing -l app=v2
    741  k get pods --show-annotations
  742  k lael -h
  743  k label -h
  744  kubectl label pods app-
  745  k get pods -l app| kubectl label pods app-
  746  kubectl label pods nginx{1..3} app-
  747  kubectl label po -l app app-
  748  k explain po.spec
  749  k desc po
  750  k desc pod
  751  k explain po.spec | grep affinity
  752  k annotate -h
  753  kubectl annotate po nginx{1..3} description='my description'
  754  k annotate po nginx{1..3} description-
  755  k annotate pods nginx1 --list
  756  kubectl annotate po nginx{1..3} description='my description'
  757  k annotate pods nginx1 --list
  758  k annotate po nginx{1..3} description-
  759  k create deployment nginx --image=nginx:1.18.0 -h
  760  k create deployment nginx --image=nginx:1.18.0 -r 2 --port=80 $do
  761  k create deployment nginx --image=nginx:1.18.0 -r 2 --port=80 $do > deploynginx.yaml
  762  vim nginx.yaml
  763  vim deploynginx.yaml
  764  k create -f deploynginx.yaml
  765  vim deploynginx.yaml
  766  k get rs
  767  k get deploy,rs
  768  k get replicaset.apps/nginx-79fccc485 -o yaml
  769  k get deploy,rs,pod
  770  k rollout deployment.apps/nginx
  771  k rollout -h
  772  k rollout status deployment.apps/nginx
    773  k rollout history deployment.apps/nginx
  774  vim deploynginx.yaml
  775  k apply -f deploynginx.yaml
  776  k get deployments/nginx -o wide
  777  k rollout history deployment.apps/nginx
  778  k rollout undo deployment.apps/nginx
  779  k rollout history deployment.apps/nginx
  780  k get deployments/nginx -o wide
  781  vim deploynginx.yaml
  782  k apply -f deploynginx.yaml
  783  k get deployments/nginx -o wide
  784  k decribe deployments/nginx
  785  k describe deployments/nginx
  786  k logs deployments/nginx
  787  k rollout status deployment.apps/nginx
  788  k get deploy,rs,pod
  789  k get deploy,rs,pod nginx
  790  k get deploy,rs,pod | grep nginx
  791  k rollout history deployment.apps/nginx
  792  k rollout undo deployment.apps/nginx -h
  793  k rollout undo deployment.apps/nginx --to-revision=2
  794  k describe deploy,rs,pod | grep nginx
  795  k get deploy,rs,pod | grep nginx
  796  k describe deployment.apps/nginx | grep 1.19.8
  797  k describe pod/nginx-7b7fdfb94b-khhcx | grep 1.19.8
  798  k rollout history deployment.apps/nginx
  799  k rollout history deployment.apps/nginx -h
  800  k rollout history deployment.apps/nginx --revision=4
  801  k scale deployment -h
  802  k scale deployment --replicas=5
  803  k scale deployment deployment.apps/nginx --replicas=5
  804  k scale deployment.apps/nginx --replicas=5
    805  k autoscale -h
  806  k autoscale deployment.apps/nginx --min=5 --max=10 --cpu-percent=80
  807  k rollout deployment.apps/nginx pause
  808  k rollout deployment.apps/nginx pause -h
  809  k rollout pause deployment.apps/nginx
  810  k rollout pause deployment.apps/nginx -h
  811  k get deploy,hpa
  812  k delete deployment.apps/nginx
  813  k delete horizontalpodautoscaler.autoscaling/nginx
  814  k create deploy lb -h
  815  export do='--dry-run=client -o yaml'
  816  export now='--force --grace-period 0'
  817  k create job -h
  818  k create job pi --image=perl -- perl -Mbignum=bpi -wle 'print bpi(2000)'"
  819  k create job pi --image=perl -- perl -Mbignum=bpi -wle 'print bpi(2000)'" --do
  820  k get jobs
  821  k create job pi --image=perl -- perl -Mbignum=bpi -wle 'print bpi(2000)'" --do
  822  k create job pi --image=perl -- perl -Mbignum=bpi -wle 'print bpi(2000)'" $do
  823  k create job pi --image=perl -- perl -Mbignum=bpi -wle 'print bpi(2000)'
  824  k get job
  825  k get job -o wide
  826  k get job -w
  827  k get job
  828  k get job -w
  829  k get job -h
  830  k get job -w
  831  k get pods
  832  k describe po pi-hlx7d | tail -n 10
  833  k logs pi-hlx7d
  834  k create job job1 -h
  835  k create job job1 --image=busybox -- 'echo hello;sleep 30;echo world' $do
  836  k get job.batch1/job1 -o yaml
    837  k get job job1 -o yaml
  838  k delete job job1
  839  k create job job1 --image=busybox -- 'echo hello;sleep 30;echo world'
  840  k get job job1 -o yaml
  841  k get pods
  842  k logs job1
  843  k logs job1-46q2g
  844  k describe pod job1-46q2g
  845  k get job job1 -o yaml > job1.yaml
  846  vim job1.yaml
  847  k delete job job1
  848  k create -f job1.yaml
  849  k create job job1 --image=busybox $do > job1.yaml
  850  vim job1.yaml
  851  k create -f job1.yaml
  852  k get jobs.batch
  853  k get pod
  854  k get pod | grep job
  855  k logs job1-jswrw
  856  vim job1.yaml
  857  k delete job job1
  858  k create -f job1.yaml
  859  k get pod | grep job
  860  k logs job1
  861  k logs job.batch/job1
  862  vim job1.yaml
  863  k get pod | grep job
  864  k get job job1 -o wide
  865  k logs job1-6b4wj
  866  k logs job.batch/job1
  867  vim job1.yaml
  868  k explain job.spec
    869  k explain job.spec | grep activeDeadlineSeconds
  870  k create job job2 -h
  871  vim job1.yaml
  872  k delete jobs.batch job1
  873  k create -f job1.yaml
  874  k get jobs.batch job1
  875  k delete jobs.batch job1
  876  vim job1.yaml
  877  k create -f job1.yaml
  878  k get jobs.batch job1
  879  k get jobs.batch job1 -w
  880  k get jobs.batch job1
  881  k describe jobs.batch job1
  882  k describe jobs.batch job1 | tail -n 10
  883  k create cm -h
  884  k create cm config --from-literal=foo=lala --from-literal=foo2=lolo
  885  k get cm config
  886  k show cm
  887  k get cm config -o wide
  888  k get cm config -o yaml
  889  k describe cm config
  890  echo -e "foo3=lili\nfoo4=lele" > config.txt
  891  k create cm config --from-file=config.txt
  892  k create cm config2 --from-file=config.txt
  893  k get cm config2
  894  k get cm config2 -o yaml
  895  --from-env-file=
  896  echo -e "var1=val1\n# this is a comment\n\nvar2=val2\n#anothercomment" > config.env
  897  k create cm config3 --from-env-file=config.env
  898  k get cm config3 -o yaml
  899  k create cm options --from-literal=var5=val5
  900  k get cm options -o yaml
    901  k create pod secpod --image=nginx $do
  902  k run secpod --image=nginx $do
  903  k run secpod --image=nginx $do > secpod.yaml
  904  vim secpod.yaml
  905  k explain po.spec | grep env
  906  k list po.spec
  907  k explain po.spec
  908  vim secpod.yaml
  909  k get cm
  910  vim secpod.yaml
  911  k get cm options
  912  k get cm options -o yaml
  913  vim secpod.yaml
  914  k create -f secpod.yaml
  915  k exec secpod -it -- echo option
  916  k exec secpod -it -- /bin/sh
  917  k exec secpod -it -- echo $option
  918  k exec secpod -it -- /bin/sh -c 'echo $option'
  919  k create cm anotherone --from-literal=var6=val6 --from-literal=var7=val7
  920  k get pods
  921  k run secpod2 --image=nginx $do > secpod2.yaml
  922  vim secpod2.yaml
  923  k exec secpod -it -- /bin/sh -c 'env | grep var'
  924  k exec secpod2 -it -- /bin/sh -c 'env | grep var'
  925  k create -f secpod2.yaml
  926  k exec secpod2 -it -- /bin/sh -c 'env | grep var'
  927  k exec secpod2 -it -- /bin/sh -c 'echo var6'
  928  k exec secpod2 -it -- /bin/sh -c 'echo $var6'
  929  k exec secpod2 -it -- /bin/sh -c 'echo $var7'
  930  k create cm cmvolume --from-literal=var8=val8 --from-literal=var9=val9
  931  cp secpod2.yaml secpod3.yaml
  932  vim secpod3.yaml
    933  k create -f secpod3.yaml
  934  vim secpod3.yaml
  935  k get pods secpod3
  936  k create -f secpod3.yaml
  937  vim secpod3.yaml
  938  k create -f secpod3.yaml
  939  k exec secpod3 -it -- /bin/bash -c 'ls /etc/lala'
  940  k exec secpod3 -it -- /bin/bash -c 'ls -l /etc/lala'
  941  k run seccon  --image=nginx $do > seccon.yaml
  942  vim seccon.yaml
  943  k create secret -h
  944  k create secret generic -h
  945  k create secret generic mysecret --from-literal=password=mypass
  946  echo -n admin > username
  947  k create secret generic --from-file=username
  948  ls
  949  k create secret generic mysecret2 --from-file=username
  950  k get secret mysecret2
  951  k get secret mysecret2 -o yaml
  952  echo YWRtaW4= | base64 -d
  953  k get pods
  954  k run secpod4 --image=nginx -h
  955  k run secpod4 --image=nginx $do > secpod4.yaml
  956  vim secpod4.yaml
  957  k create -f secpod4.yaml
  958  k get pods | grep sec
  959  k exec secpod4 -it -- /bin/bash
  960  k delete pod secpod4
  961  k get serviceaccounts
  962  k get serviceaccounts -A
  963  k create sa -h
  964  k create sa myuser
    965  k run --image=nginx $do > sa.yaml
  966  k run sapod --image=nginx $do > sa.yaml
  967  mv sa.yaml sapod.yaml
  968  vim sapod.yaml
  969  k create -f sapod.yaml
  970  k describe pod sapod | grep account
  971  k describe pod sapod | grep serviceaccount
  972  k describe pod sapod | grep sa
  973  k describe pod sapod
  974  k get secret
  975  ls | grep histpry
  976  ls | grep history
  977  history > history.out
```
