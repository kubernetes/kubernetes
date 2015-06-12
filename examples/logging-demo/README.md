# Elasticsearch/Kibana Logging Demonstration
This directory contains two [pod](../../docs/pods.md) specifications which can be used as synthetic
logging sources. The pod specification in [synthetic_0_25lps.yaml](synthetic_0_25lps.yaml)
describes a pod that just emits a log message once every 4 seconds:
```
# This pod specification creates an instance of a synthetic logger. The logger
# is simply a program that writes out the hostname of the pod, a count which increments
# by one on each iteration (to help notice missing log enteries) and the date using
# a long format (RFC-3339) to nano-second precision. This program logs at a frequency
# of 0.25 lines per second. The shellscript program is given directly to bash as -c argument
# and could have been written out as:
#   i="0"
#   while true
#   do
#     echo -n "`hostname`: $i: "
#     date --rfc-3339 ns
#     sleep 4
#     i=$[$i+1]
#   done
apiVersion: v1
kind: Pod
metadata:
  labels:
    name: synth-logging-source
  name: synthetic-logger-0.25lps-pod
spec:
  containers:
  - args:
    - bash
    - -c
    - 'i="0"; while true; do echo -n "`hostname`: $i: "; date --rfc-3339 ns; sleep
      4; i=$[$i+1]; done'
    image: ubuntu:14.04
    name: synth-lgr
```

The other YAML file [synthetic_10lps.yaml](synthetic_10lps.yaml) specifies a similar synthetic logger that emits 10 log messages every second. To run both synthetic loggers:
```
$ make up
../../cluster/kubectl.sh create -f synthetic_0_25lps.yaml
pods/synthetic-logger-0.25lps-pod
../../cluster/kubectl.sh create -f synthetic_10lps.yaml
pods/synthetic-logger-10lps-pod
```

Visiting the Kibana dashboard should make it clear that logs are being collected from the two synthetic loggers:
![Synthetic loggers](synth-logger.png)

You can report the running pods, [replication controllers](../../docs/replication-controller.md), and [services](../../docs/services.md) with another Makefile rule:
```
$ make get
../../cluster/kubectl.sh get pods
NAME                                           READY     REASON    RESTARTS   AGE
elasticsearch-logging-v1-gzknt                 1/1       Running   0          11m
elasticsearch-logging-v1-swgzc                 1/1       Running   0          11m
fluentd-elasticsearch-kubernetes-minion-1rtv   1/1       Running   0          11m
fluentd-elasticsearch-kubernetes-minion-6bob   1/1       Running   0          10m
fluentd-elasticsearch-kubernetes-minion-98g3   1/1       Running   0          10m
fluentd-elasticsearch-kubernetes-minion-qduc   1/1       Running   0          10m
kibana-logging-v1-1w44h                        1/1       Running   0          11m
kube-dns-v3-i8u9s                              3/3       Running   0          11m
monitoring-heapster-v1-mles8                   0/1       Running   11         11m
synthetic-logger-0.25lps-pod                   1/1       Running   0          42s
synthetic-logger-10lps-pod                     1/1       Running   0          41s
../../cluster/kubectl.sh get replicationControllers
CONTROLLER                 CONTAINER(S)            IMAGE(S)                                         SELECTOR                                   REPLICAS
elasticsearch-logging-v1   elasticsearch-logging   gcr.io/google_containers/elasticsearch:1.4       k8s-app=elasticsearch-logging,version=v1   2
kibana-logging-v1          kibana-logging          gcr.io/google_containers/kibana:1.3              k8s-app=kibana-logging,version=v1          1
kube-dns-v3                etcd                    gcr.io/google_containers/etcd:2.0.9              k8s-app=kube-dns,version=v3                1
                           kube2sky                gcr.io/google_containers/kube2sky:1.9                                                       
                           skydns                  gcr.io/google_containers/skydns:2015-03-11-001                                              
monitoring-heapster-v1     heapster                gcr.io/google_containers/heapster:v0.13.0        k8s-app=heapster,version=v1                1
../../cluster/kubectl.sh get services
NAME                    LABELS                                                                                              SELECTOR                        IP(S)          PORT(S)
elasticsearch-logging   k8s-app=elasticsearch-logging,kubernetes.io/cluster-service=true,kubernetes.io/name=Elasticsearch   k8s-app=elasticsearch-logging   10.0.145.125   9200/TCP
kibana-logging          k8s-app=kibana-logging,kubernetes.io/cluster-service=true,kubernetes.io/name=Kibana                 k8s-app=kibana-logging          10.0.189.192   5601/TCP
kube-dns                k8s-app=kube-dns,kubernetes.io/cluster-service=true,kubernetes.io/name=KubeDNS                      k8s-app=kube-dns                10.0.0.10      53/UDP
                                                                                                                                                                           53/TCP
kubernetes              component=apiserver,provider=kubernetes                                                             <none>                          10.0.0.1       443/TCP
```

The `net` rule in the Makefile will report information about the Elasticsearch and Kibana services including the public IP addresses of each service.
```
$ make net
../../cluster/kubectl.sh get services elasticsearch-logging -o json
{
    "kind": "Service",
    "apiVersion": "v1",
    "metadata": {
        "name": "elasticsearch-logging",
        "namespace": "default",
        "selfLink": "/api/v1/namespaces/default/services/elasticsearch-logging",
        "uid": "e056e116-0fb4-11e5-9243-42010af0d13a",
        "resourceVersion": "23",
        "creationTimestamp": "2015-06-10T21:08:43Z",
        "labels": {
            "k8s-app": "elasticsearch-logging",
            "kubernetes.io/cluster-service": "true",
            "kubernetes.io/name": "Elasticsearch"
        }
    },
    "spec": {
        "ports": [
            {
                "protocol": "TCP",
                "port": 9200,
                "targetPort": "es-port",
                "nodePort": 0
            }
        ],
        "selector": {
            "k8s-app": "elasticsearch-logging"
        },
        "clusterIP": "10.0.145.125",
        "type": "ClusterIP",
        "sessionAffinity": "None"
    },
    "status": {
        "loadBalancer": {}
    }
}
../../cluster/kubectl.sh get services kibana-logging -o json
{
    "kind": "Service",
    "apiVersion": "v1",
    "metadata": {
        "name": "kibana-logging",
        "namespace": "default",
        "selfLink": "/api/v1/namespaces/default/services/kibana-logging",
        "uid": "e05c7dae-0fb4-11e5-9243-42010af0d13a",
        "resourceVersion": "30",
        "creationTimestamp": "2015-06-10T21:08:43Z",
        "labels": {
            "k8s-app": "kibana-logging",
            "kubernetes.io/cluster-service": "true",
            "kubernetes.io/name": "Kibana"
        }
    },
    "spec": {
        "ports": [
            {
                "protocol": "TCP",
                "port": 5601,
                "targetPort": "kibana-port",
                "nodePort": 0
            }
        ],
        "selector": {
            "k8s-app": "kibana-logging"
        },
        "clusterIP": "10.0.189.192",
        "type": "ClusterIP",
        "sessionAffinity": "None"
    },
    "status": {
        "loadBalancer": {}
    }
}
```
To find the URLs to access the Elasticsearch and Kibana viewer,
```
$ kubectl cluster-info
Kubernetes master is running at https://104.154.60.226
Elasticsearch is running at https://104.154.60.226/api/v1beta3/proxy/namespaces/default/services/elasticsearch-logging
Kibana is running at https://104.154.60.226/api/v1beta3/proxy/namespaces/default/services/kibana-logging
KubeDNS is running at https://104.154.60.226/api/v1beta3/proxy/namespaces/default/services/kube-dns
```

To find the user name and password to access the URLs,
```
$ kubectl config view 
apiVersion: v1
clusters:
- cluster:
    certificate-authority-data: REDACTED
    server: https://104.154.60.226 
  name: lithe-cocoa-92103_kubernetes
contexts:
- context:
    cluster: lithe-cocoa-92103_kubernetes
    user: lithe-cocoa-92103_kubernetes
  name: lithe-cocoa-92103_kubernetes
current-context: lithe-cocoa-92103_kubernetes
kind: Config
preferences: {}
users:
- name: lithe-cocoa-92103_kubernetes
  user:
    client-certificate-data: REDACTED
    client-key-data: REDACTED
    token: 65rZW78y8HxmXXtSXuUw9DbP4FLjHi4b
- name: lithe-cocoa-92103_kubernetes-basic-auth
  user:
    password: h5M0FtVXXflBSdI7
    username: admin
```

Access the Elasticsearch service at URL `https://104.154.60.226/api/v1/proxy/namespaces/default/services/elasticsearch-logging`, use the user name 'admin' and password 'h5M0FtVXXflBSdI7',
```
{
  "status" : 200,
  "name" : "Major Mapleleaf",
  "cluster_name" : "kubernetes_logging",
  "version" : {
    "number" : "1.4.4",
    "build_hash" : "c88f77ffc81301dfa9dfd81ca2232f09588bd512",
    "build_timestamp" : "2015-02-19T13:05:36Z",
    "build_snapshot" : false,
    "lucene_version" : "4.10.3"
  },
  "tagline" : "You Know, for Search"
}
```
Visiting the URL `https://104.154.60.226/api/v1/proxy/namespaces/default/services/kibana-logging` should show the Kibana viewer for the logging information stored in the Elasticsearch service.


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/logging-demo/README.md?pixel)]()
