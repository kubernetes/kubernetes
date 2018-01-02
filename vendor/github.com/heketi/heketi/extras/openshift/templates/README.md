# Create a Heketi service in OpenShift
> NOTE: This template file places the database in an _EmptyDir_ volume.  You will need to adjust accordingly if you would like the database to be on reliable persistent storage.

* Register template with OpenShift

```
oc create -f heketi.json
```

* Note the number of parameters which need to be set.  Currently only _NAME_
  needs to be set.

```
oc process --parameters heketi
```

* Deploy a Heketi service

Here is an example of how to deploy Heketi

```
oc process heketi -v NAME=myheketiservice \
     HEKETI_KUBE_NAMESPACE=test \
     HEKETI_KUBE_APIHOST='https://192.168.10.90:8443' \
     HEKETI_KUBE_INSECURE=y \
     HEKETI_KUBE_USER=test-admin \
     HEKETI_KUBE_PASSWORD=admin | oc create -f -
```

* Note service

```
oc status
```

* Send a _hello_ command to service

```
curl http://<ip of service>:<port>/hello
```

* For example

```
$ oc project
Using project "gluster" on server "https://192.168.10.90:8443".

$ oc create -f heketi-template.json 
template "heketi" created

$ oc process heketi -v NAME=ams \
>      HEKETI_KUBE_NAMESPACE=gluster \
>      HEKETI_KUBE_APIHOST='https://192.168.10.90:8443' \
>      HEKETI_KUBE_INSECURE=y \
>      HEKETI_KUBE_USER=test-admin \
>      HEKETI_KUBE_PASSWORD=admin | oc create -f -
service "ams" created
deploymentconfig "ams" created

$ oc status
In project gluster on server https://192.168.10.90:8443

svc/ams - 172.30.244.79:8080
  dc/ams deploys docker.io/heketi/heketi:dev
    deployment #1 pending 5 seconds ago

View details with 'oc describe <resource>/<name>' or list everything with 'oc get all'.

$ oc get pods -o wide
NAME           READY     STATUS              RESTARTS   AGE       NODE
ams-1-bed48    0/1       ContainerCreating   0          8s        openshift-node-1
ams-1-deploy   1/1       Running             0          1m        openshift-node-2

<< Wait until the container is running, then... >>

$ curl http://172.30.244.79:8080/hello
HelloWorld from GlusterFS Application
```



