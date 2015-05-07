# Elasticsearch for Kubernetes

This directory contains the source for a Docker image that creates an instance
of [Elasticsearch](https://www.elastic.co/products/elasticsearch) 1.5.2 which can 
be used to automatically form clusters when used
with replication controllers. This will not work with the library Elasticsearch image
because multicast discovery will not find the other pod IPs needed to form a cluster. This
image detects other Elasticsearch pods running in a specified namespace with a given
label selector. The detected instances are used to form a list of peer hosts which
are used as part of the unicast discovery mechansim for Elasticsearch. The detection
of the peer nodes is done by a program which communicates with the Kubernetes API
server to get a list of matching Elasticsearch pods. To enable authenticated
communication this image needs a secret to be mounted at `/etc/apiserver-secret`
with the basic authentication username and password.

Here is an example replication controller specification that creates 4 instances of Elasticsearch which is in the file
[music-rc.yaml](music-rc.yaml).
```
apiVersion: v1beta3
kind: ReplicationController
metadata:
  labels:
    name: music-db
    namespace: mytunes
  name: music-db
spec:
  replicas: 4
  selector:
    name: music-db
  template:
    metadata:
      labels:
         name: music-db
    spec:
      containers:
      - name: es
        image: kubernetes/elasticsearch:1.0
        env:
          - name: "CLUSTER_NAME"
            value: "mytunes-db"
          - name: "SELECTOR"
            value: "name=music-db"
          - name: "NAMESPACE"
            value: "mytunes"
        ports:
        - name: es
          containerPort: 9200
        - name: es-transport
          containerPort: 9300
        volumeMounts:
        - name: apiserver-secret
          mountPath: /etc/apiserver-secret
          readOnly: true
      volumes:
      - name: apiserver-secret
        secret:
          secretName: apiserver-secret
```
The `CLUSTER_NAME` variable gives a name to the cluster and allows multiple separate clusters to
exist in the same namespace.
The `SELECTOR` variable should be set to a label query that identifies the Elasticsearch
nodes that should participate in this cluster. For our example we specify `name=music-db` to
match all pods that have the label `name` set to the value `music-db`.
The `NAMESPACE` variable identifies the namespace
to be used to search for Elasticsearch pods and this should be the same as the namespace specified
for the replication controller (in this case `mytunes`). 

Before creating pods with the replication controller a secret containing the bearer authentication token
should be set up. A template is provided in the file [apiserver-secret.yaml](apiserver-secret.yaml):
```
apiVersion: v1beta3
kind: Secret
metadata:
  name: apiserver-secret
  namespace: NAMESPACE
data:
  token: "TOKEN"

```
Replace `NAMESPACE` with the actual namespace to be used and `TOKEN` with the basic64 encoded
versions of the bearer token reported by `kubectl config view` e.g.
```
$ kubectl config view
...
- name: kubernetes-logging_kubernetes-basic-auth
...
  token: yGlDcMvSZPX4PyP0Q5bHgAYgi1iyEHv2
 ...   
$ echo yGlDcMvSZPX4PyP0Q5bHgAYgi1iyEHv2 | base64
eUdsRGNNdlNaUFg0UHlQMFE1YkhnQVlnaTFpeUVIdjIK=

```
resulting in the file:
```
apiVersion: v1beta3
kind: Secret
metadata:
  name: apiserver-secret
  namespace: mytunes
data:
  token: "eUdsRGNNdlNaUFg0UHlQMFE1YkhnQVlnaTFpeUVIdjIK="

```
which can be used to create the secret in your namespace:
```
kubectl create -f apiserver-secret.yaml --namespace=mytunes
secrets/apiserver-secret

```
Now you are ready to create the replication controller which will then create the pods:
```
$ kubectl create -f music-rc.yaml --namespace=mytunes
replicationcontrollers/music-db

```
It's also useful to have a service with an external load balancer for accessing the Elasticsearch
cluster which can be found in the file [music-service.yaml](music-service.yaml).
```
apiVersion: v1beta3
kind: Service
metadata:
  name: music-server
  namespace: mytunes
  labels:
    name: music-db
spec:
  selector:
    name: music-db
  ports:
  - name: db
    port: 9200
    targetPort: es
  createExternalLoadBalancer: true
```
Let's create the service with an external load balancer:
```
$ kubectl create -f music-service.yaml --namespace=mytunes
services/music-server

```
Let's see what we've got:
```
$ kubectl get pods,rc,services,secrets --namespace=mytunes

POD              IP            CONTAINER(S)   IMAGE(S)                       HOST                                     LABELS          STATUS    CREATED      MESSAGE
music-db-0fwsu   10.244.2.48                                                 kubernetes-minion-m49b/104.197.35.221    name=music-db   Running   6 minutes    
                               es             kubernetes/elasticsearch:1.0                                                            Running   29 seconds   
music-db-5pc2e   10.244.0.24                                                 kubernetes-minion-3c8c/146.148.41.184    name=music-db   Running   6 minutes    
                               es             kubernetes/elasticsearch:1.0                                                            Running   6 minutes    
music-db-bjqmv   10.244.3.31                                                 kubernetes-minion-zey5/104.154.59.10     name=music-db   Running   6 minutes    
                               es             kubernetes/elasticsearch:1.0                                                            Running   19 seconds   
music-db-swtrs   10.244.1.37                                                 kubernetes-minion-f9dw/130.211.159.230   name=music-db   Running   6 minutes    
                               es             kubernetes/elasticsearch:1.0                                                            Running   6 minutes    
CONTROLLER   CONTAINER(S)   IMAGE(S)                       SELECTOR        REPLICAS
music-db     es             kubernetes/elasticsearch:1.0   name=music-db   4
NAME           LABELS          SELECTOR        IP(S)            PORT(S)
music-server   name=music-db   name=music-db   10.0.138.61      9200/TCP
                                               104.197.12.157   
NAME               TYPE      DATA
apiserver-secret   Opaque    2
```
This shows 4 instances of Elasticsearch running. After making sure that port 9200 is accessible for this cluster (e.g. using a firewall rule for GCE) we can make queries via the service which will be fielded by the matching Elasticsearch pods.
```
$ curl 104.197.12.157:9200
{
  "status" : 200,
  "name" : "Warpath",
  "cluster_name" : "mytunes-db",
  "version" : {
    "number" : "1.5.2",
    "build_hash" : "62ff9868b4c8a0c45860bebb259e21980778ab1c",
    "build_timestamp" : "2015-04-27T09:21:06Z",
    "build_snapshot" : false,
    "lucene_version" : "4.10.4"
  },
  "tagline" : "You Know, for Search"
}
$ curl 104.197.12.157:9200
{
  "status" : 200,
  "name" : "Callisto",
  "cluster_name" : "mytunes-db",
  "version" : {
    "number" : "1.5.2",
    "build_hash" : "62ff9868b4c8a0c45860bebb259e21980778ab1c",
    "build_timestamp" : "2015-04-27T09:21:06Z",
    "build_snapshot" : false,
    "lucene_version" : "4.10.4"
  },
  "tagline" : "You Know, for Search"
}
```
We can query the nodes to confirm that an Elasticsearch cluster has been formed.
```
$ curl 104.197.12.157:9200/_nodes?pretty=true
{
  "cluster_name" : "mytunes-db",
  "nodes" : {
    "u-KrvywFQmyaH5BulSclsA" : {
      "name" : "Jonas Harrow",
...
        "discovery" : {
          "zen" : {
            "ping" : {
              "unicast" : {
                "hosts" : [ "10.244.2.48", "10.244.0.24", "10.244.3.31", "10.244.1.37" ]
              },
...
      "name" : "Warpath",
...
        "discovery" : {
          "zen" : {
            "ping" : {
              "unicast" : {
                "hosts" : [ "10.244.2.48", "10.244.0.24", "10.244.3.31", "10.244.1.37" ]
              },
...
        "name" : "Callisto",
...
        "discovery" : {
          "zen" : {
            "ping" : {
              "unicast" : {
                "hosts" : [ "10.244.2.48", "10.244.0.24", "10.244.3.31", "10.244.1.37" ]
              },
...
      "name" : "Vapor",
...
        "discovery" : {
          "zen" : {
            "ping" : {
              "unicast" : {
                "hosts" : [ "10.244.2.48", "10.244.0.24", "10.244.3.31", "10.244.1.37" ]
...
```
Let's ramp up the number of Elasticsearch nodes from 4 to 10:
```
$ kubectl resize --replicas=10 replicationcontrollers music-db --namespace=mytunes
resized
$ kubectl get pods --namespace=mytunes
POD              IP            CONTAINER(S)   IMAGE(S)                       HOST                                     LABELS          STATUS    CREATED      MESSAGE
music-db-0fwsu   10.244.2.48                                                 kubernetes-minion-m49b/104.197.35.221    name=music-db   Running   33 minutes   
                               es             kubernetes/elasticsearch:1.0                                                            Running   26 minutes   
music-db-2erje   10.244.2.50                                                 kubernetes-minion-m49b/104.197.35.221    name=music-db   Running   48 seconds   
                               es             kubernetes/elasticsearch:1.0                                                            Running   46 seconds   
music-db-5pc2e   10.244.0.24                                                 kubernetes-minion-3c8c/146.148.41.184    name=music-db   Running   33 minutes   
                               es             kubernetes/elasticsearch:1.0                                                            Running   32 minutes   
music-db-8rkvp   10.244.3.33                                                 kubernetes-minion-zey5/104.154.59.10     name=music-db   Running   48 seconds   
                               es             kubernetes/elasticsearch:1.0                                                            Running   46 seconds   
music-db-bjqmv   10.244.3.31                                                 kubernetes-minion-zey5/104.154.59.10     name=music-db   Running   33 minutes   
                               es             kubernetes/elasticsearch:1.0                                                            Running   26 minutes   
music-db-efc46   10.244.2.49                                                 kubernetes-minion-m49b/104.197.35.221    name=music-db   Running   48 seconds   
                               es             kubernetes/elasticsearch:1.0                                                            Running   46 seconds   
music-db-fhqyg   10.244.0.25                                                 kubernetes-minion-3c8c/146.148.41.184    name=music-db   Running   48 seconds   
                               es             kubernetes/elasticsearch:1.0                                                            Running   47 seconds   
music-db-guxe4   10.244.3.32                                                 kubernetes-minion-zey5/104.154.59.10     name=music-db   Running   48 seconds   
                               es             kubernetes/elasticsearch:1.0                                                            Running   46 seconds   
music-db-pbiq1   10.244.1.38                                                 kubernetes-minion-f9dw/130.211.159.230   name=music-db   Running   48 seconds   
                               es             kubernetes/elasticsearch:1.0                                                            Running   47 seconds   
music-db-swtrs   10.244.1.37                                                 kubernetes-minion-f9dw/130.211.159.230   name=music-db   Running   33 minutes   
                               es             kubernetes/elasticsearch:1.0                                                            Running   32 minutes 

```
Let's check to make sure that these 10 nodes are part of the same Elasticsearch cluster:
```
$ curl 104.197.12.157:9200/_nodes?pretty=true | grep name
"cluster_name" : "mytunes-db",
      "name" : "Killraven",
        "name" : "Killraven",
          "name" : "mytunes-db"
        "vm_name" : "OpenJDK 64-Bit Server VM",
          "name" : "eth0",
      "name" : "Tefral the Surveyor",
        "name" : "Tefral the Surveyor",
          "name" : "mytunes-db"
        "vm_name" : "OpenJDK 64-Bit Server VM",
          "name" : "eth0",
      "name" : "Jonas Harrow",
        "name" : "Jonas Harrow",
          "name" : "mytunes-db"
        "vm_name" : "OpenJDK 64-Bit Server VM",
          "name" : "eth0",
      "name" : "Warpath",
        "name" : "Warpath",
          "name" : "mytunes-db"
        "vm_name" : "OpenJDK 64-Bit Server VM",
          "name" : "eth0",
      "name" : "Brute I",
        "name" : "Brute I",
          "name" : "mytunes-db"
        "vm_name" : "OpenJDK 64-Bit Server VM",
          "name" : "eth0",
      "name" : "Callisto",
        "name" : "Callisto",
          "name" : "mytunes-db"
        "vm_name" : "OpenJDK 64-Bit Server VM",
          "name" : "eth0",
      "name" : "Vapor",
        "name" : "Vapor",
          "name" : "mytunes-db"
        "vm_name" : "OpenJDK 64-Bit Server VM",
          "name" : "eth0",
      "name" : "Timeslip",
        "name" : "Timeslip",
          "name" : "mytunes-db"
        "vm_name" : "OpenJDK 64-Bit Server VM",
          "name" : "eth0",
      "name" : "Magik",
        "name" : "Magik",
          "name" : "mytunes-db"
        "vm_name" : "OpenJDK 64-Bit Server VM",
          "name" : "eth0",
      "name" : "Brother Voodoo",
        "name" : "Brother Voodoo",
          "name" : "mytunes-db"
        "vm_name" : "OpenJDK 64-Bit Server VM",
          "name" : "eth0",

```