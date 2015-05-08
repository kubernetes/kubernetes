# Elasticsearch/Kibana Logging Demonstration
This directory contains two pod specifications which can be used as synthetic
loggig sources. The pod specification in [synthetic_0_25lps.yaml](synthetic_0_25lps.yaml)
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
apiVersion: v1beta3
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
../../../kubectl.sh create -f synthetic_0_25lps.yaml
Running: ../../../cluster/../cluster/gce/../../_output/dockerized/bin/linux/amd64/kubectl create -f synthetic_0_25lps.yaml
synthetic-logger-0.25lps-pod
../../../kubectl.sh create -f synthetic_10lps.yaml
Running: ../../../cluster/../cluster/gce/../../_output/dockerized/bin/linux/amd64/kubectl create -f synthetic_10lps.yaml
synthetic-logger-10lps-pod

```

Visiting the Kibana dashboard should make it clear that logs are being collected from the two synthetic loggers:
![Synthetic loggers](synth-logger.png)

You can report the running pods, replication controllers and services with another Makefile rule:
```
$ make get
../../../kubectl.sh get pods
Running: ../../../../cluster/gce/../../_output/dockerized/bin/linux/amd64/kubectl get pods
POD                                            IP            CONTAINER(S)              IMAGE(S)                                                                            HOST                                    LABELS                                                              STATUS    CREATED      MESSAGE
elasticsearch-logging-f0smz                    10.244.2.3                                                                                                                  kubernetes-minion-ilqx/104.197.8.214    kubernetes.io/cluster-service=true,name=elasticsearch-logging       Running   5 hours      
                                                             elasticsearch-logging     gcr.io/google_containers/elasticsearch:1.0                                                                                                                                                      Running   5 hours      
etcd-server-kubernetes-master                                                                                                                                              kubernetes-master/                      <none>                                                              Running   5 hours      
                                                             etcd-container            gcr.io/google_containers/etcd:2.0.9                                                                                                                                                             Running   5 hours      
fluentd-elasticsearch-kubernetes-minion-7s1y   10.244.0.2                                                                                                                  kubernetes-minion-7s1y/23.236.54.97     <none>                                                              Running   5 hours      
                                                             fluentd-elasticsearch     gcr.io/google_containers/fluentd-elasticsearch:1.5                                                                                                                                              Running   5 hours      
fluentd-elasticsearch-kubernetes-minion-cfs6   10.244.1.2                                                                                                                  kubernetes-minion-cfs6/104.154.61.231   <none>                                                              Running   5 hours      
                                                             fluentd-elasticsearch     gcr.io/google_containers/fluentd-elasticsearch:1.5                                                                                                                                              Running   5 hours      
fluentd-elasticsearch-kubernetes-minion-ilqx   10.244.2.2                                                                                                                  kubernetes-minion-ilqx/104.197.8.214    <none>                                                              Running   5 hours      
                                                             fluentd-elasticsearch     gcr.io/google_containers/fluentd-elasticsearch:1.5                                                                                                                                              Running   5 hours      
fluentd-elasticsearch-kubernetes-minion-x8gx   10.244.3.2                                                                                                                  kubernetes-minion-x8gx/104.154.47.83    <none>                                                              Running   5 hours      
                                                             fluentd-elasticsearch     gcr.io/google_containers/fluentd-elasticsearch:1.5                                                                                                                                              Running   5 hours      
kibana-logging-cwe0b                           10.244.1.3                                                                                                                  kubernetes-minion-cfs6/104.154.61.231   kubernetes.io/cluster-service=true,name=kibana-logging              Running   5 hours      
                                                             kibana-logging            gcr.io/google_containers/kibana:1.2                                                                                                                                                             Running   5 hours      
kube-apiserver-kubernetes-master                                                                                                                                           kubernetes-master/                      <none>                                                              Running   5 hours      
                                                             kube-apiserver            gcr.io/google_containers/kube-apiserver:f0c332fc2582927ec27d24965572d4b0                                                                                                                        Running   5 hours      
kube-controller-manager-kubernetes-master                                                                                                                                  kubernetes-master/                      <none>                                                              Running   5 hours      
                                                             kube-controller-manager   gcr.io/google_containers/kube-controller-manager:6729154dfd4e2a19752bdf9ceff8464c                                                                                                               Running   5 hours      
kube-dns-swd4n                                 10.244.3.5                                                                                                                  kubernetes-minion-x8gx/104.154.47.83    k8s-app=kube-dns,kubernetes.io/cluster-service=true,name=kube-dns   Running   5 hours      
                                                             kube2sky                  gcr.io/google_containers/kube2sky:1.2                                                                                                                                                           Running   5 hours      
                                                             etcd                      quay.io/coreos/etcd:v2.0.3                                                                                                                                                                      Running   5 hours      
                                                             skydns                    gcr.io/google_containers/skydns:2015-03-11-001                                                                                                                                                  Running   5 hours      
kube-scheduler-kubernetes-master                                                                                                                                           kubernetes-master/                      <none>                                                              Running   5 hours      
                                                             kube-scheduler            gcr.io/google_containers/kube-scheduler:ec9d2092f754211cc5ab3a5162c05fc1                                                                                                                        Running   5 hours      
monitoring-heapster-controller-zpjj1           10.244.3.3                                                                                                                  kubernetes-minion-x8gx/104.154.47.83    kubernetes.io/cluster-service=true,name=heapster                    Running   5 hours      
                                                             heapster                  gcr.io/google_containers/heapster:v0.10.0                                                                                                                                                       Running   5 hours      
monitoring-influx-grafana-controller-dqan4     10.244.3.4                                                                                                                  kubernetes-minion-x8gx/104.154.47.83    kubernetes.io/cluster-service=true,name=influxGrafana               Running   5 hours      
                                                             grafana                   gcr.io/google_containers/heapster_grafana:v0.6                                                                                                                                                  Running   5 hours      
                                                             influxdb                  gcr.io/google_containers/heapster_influxdb:v0.3                                                                                                                                                 Running   5 hours      
synthetic-logger-0.25lps-pod                   10.244.0.7                                                                                                                  kubernetes-minion-7s1y/23.236.54.97     name=synth-logging-source                                           Running   19 minutes   
                                                             synth-lgr                 ubuntu:14.04                                                                                                                                                                                    Running   19 minutes   
synthetic-logger-10lps-pod                     10.244.3.14                                                                                                                 kubernetes-minion-x8gx/104.154.47.83    name=synth-logging-source                                           Running   19 minutes   
                                                             synth-lgr                 ubuntu:14.04                                                                                                                                                                                    Running   19 minutes   
../../_output/local/bin/linux/amd64/kubectl get replicationControllers
CONTROLLER                             CONTAINER(S)            IMAGE(S)                                          SELECTOR                     REPLICAS
elasticsearch-logging                  elasticsearch-logging   gcr.io/google_containers/elasticsearch:1.0        name=elasticsearch-logging   1
kibana-logging                         kibana-logging          gcr.io/google_containers/kibana:1.2               name=kibana-logging          1
kube-dns                               etcd                    quay.io/coreos/etcd:v2.0.3                        k8s-app=kube-dns             1
                                       kube2sky                gcr.io/google_containers/kube2sky:1.2                                          
                                       skydns                  gcr.io/google_containers/skydns:2015-03-11-001                                 
monitoring-heapster-controller         heapster                gcr.io/google_containers/heapster:v0.10.0         name=heapster                1
monitoring-influx-grafana-controller   influxdb                gcr.io/google_containers/heapster_influxdb:v0.3   name=influxGrafana           1
                                       grafana                 gcr.io/google_containers/heapster_grafana:v0.6                                 
../../_output/local/bin/linux/amd64/kubectl get services
NAME                     LABELS                                                              SELECTOR                     IP(S)          PORT(S)
elasticsearch-logging    kubernetes.io/cluster-service=true,name=elasticsearch-logging       name=elasticsearch-logging   10.0.251.221   9200/TCP
kibana-logging           kubernetes.io/cluster-service=true,name=kibana-logging              name=kibana-logging          10.0.188.118   5601/TCP
kube-dns                 k8s-app=kube-dns,kubernetes.io/cluster-service=true,name=kube-dns   k8s-app=kube-dns             10.0.0.10      53/UDP
kubernetes               component=apiserver,provider=kubernetes                             <none>                       10.0.0.2       443/TCP
kubernetes-ro            component=apiserver,provider=kubernetes                             <none>                       10.0.0.1       80/TCP
monitoring-grafana       kubernetes.io/cluster-service=true,name=grafana                     name=influxGrafana           10.0.254.202   80/TCP
monitoring-heapster      kubernetes.io/cluster-service=true,name=heapster                    name=heapster                10.0.19.214    80/TCP
monitoring-influxdb      name=influxGrafana                                                  name=influxGrafana           10.0.198.71    80/TCP
monitoring-influxdb-ui   name=influxGrafana                                                  name=influxGrafana           10.0.109.66    80/TCP
```

The `net` rule in the Makefile will report information about the Elasticsearch and Kibana services including the public IP addresses of each service.
```
$ make net
../../../kubectl.sh get services elasticsearch-logging -o json
current-context: "lithe-cocoa-92103_kubernetes"
Running: ../../_output/local/bin/linux/amd64/kubectl get services elasticsearch-logging -o json
{
    "kind": "Service",
    "apiVersion": "v1beta3",
    "metadata": {
        "name": "elasticsearch-logging",
        "namespace": "default",
        "selfLink": "/api/v1beta3/namespaces/default/services/elasticsearch-logging",
        "uid": "9dc7290f-f358-11e4-a58e-42010af09a93",
        "resourceVersion": "28",
        "creationTimestamp": "2015-05-05T18:57:45Z",
        "labels": {
            "kubernetes.io/cluster-service": "true",
            "name": "elasticsearch-logging"
        }
    },
    "spec": {
        "ports": [
            {
                "name": "",
                "protocol": "TCP",
                "port": 9200,
                "targetPort": "es-port"
            }
        ],
        "selector": {
            "name": "elasticsearch-logging"
        },
        "portalIP": "10.0.251.221",
        "sessionAffinity": "None"
    },
    "status": {}
}
current-context: "lithe-cocoa-92103_kubernetes"
Running: ../../_output/local/bin/linux/amd64/kubectl get services kibana-logging -o json
{
    "kind": "Service",
    "apiVersion": "v1beta3",
    "metadata": {
        "name": "kibana-logging",
        "namespace": "default",
        "selfLink": "/api/v1beta3/namespaces/default/services/kibana-logging",
        "uid": "9dc6f856-f358-11e4-a58e-42010af09a93",
        "resourceVersion": "31",
        "creationTimestamp": "2015-05-05T18:57:45Z",
        "labels": {
            "kubernetes.io/cluster-service": "true",
            "name": "kibana-logging"
        }
    },
    "spec": {
        "ports": [
            {
                "name": "",
                "protocol": "TCP",
                "port": 5601,
                "targetPort": "kibana-port"
            }
        ],
        "selector": {
            "name": "kibana-logging"
        },
        "portalIP": "10.0.188.118",
        "sessionAffinity": "None"
    },
    "status": {}
}
```
To find the URLs to access the Elasticsearch and Kibana viewer,
```
$ kubectl cluster-info
Kubernetes master is running at https://130.211.122.180
elasticsearch-logging is running at https://130.211.122.180/api/v1beta3/proxy/namespaces/default/services/elasticsearch-logging
kibana-logging is running at https://130.211.122.180/api/v1beta3/proxy/namespaces/default/services/kibana-logging
kube-dns is running at https://130.211.122.180/api/v1beta3/proxy/namespaces/default/services/kube-dns
grafana is running at https://130.211.122.180/api/v1beta3/proxy/namespaces/default/services/monitoring-grafana
heapster is running at https://130.211.122.180/api/v1beta3/proxy/namespaces/default/services/monitoring-heapster
```

To find the user name and password to access the URLs,
```
$ kubectl config view 
apiVersion: v1
clusters:
- cluster:
    certificate-authority-data: REDACTED
    server: https://130.211.122.180
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

Access the Elasticsearch service at URL `https://130.211.122.180/api/v1beta3/proxy/namespaces/default/services/elasticsearch-logging`, use the user name 'admin' and password 'h5M0FtVXXflBSdI7',
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
Visiting the URL `https://130.211.122.180/api/v1beta3/proxy/namespaces/default/services/kibana-logging` should show the Kibana viewer for the logging information stored in the Elasticsearch service.
