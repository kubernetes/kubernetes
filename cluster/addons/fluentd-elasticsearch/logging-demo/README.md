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

apiVersion: v1beta1
kind: Pod
id: synthetic-logger-0.25lps-pod
desiredState:
  manifest:
    version: v1beta1
    id: synth-logger-0.25lps
    containers:
      - name: synth-lgr
        image: ubuntu:14.04
        command: ["bash", "-c", "i=\"0\"; while true; do echo -n \"`hostname`: $i: \"; date --rfc-3339 ns; sleep 4; i=$[$i+1]; done"]
labels:
  name: synth-logging-source
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
POD                                    CONTAINER(S)            IMAGE(S)                           HOST                                                            LABELS                       STATUS
7e1c7ce6-9764-11e4-898c-42010af03582   kibana-logging          kubernetes/kibana                  kubernetes-minion-3.c.kubernetes-elk.internal/130.211.129.169   name=kibana-logging          Running
synthetic-logger-0.25lps-pod           synth-lgr               ubuntu:14.04                       kubernetes-minion-2.c.kubernetes-elk.internal/146.148.41.87     name=synth-logging-source    Running
synthetic-logger-10lps-pod             synth-lgr               ubuntu:14.04                       kubernetes-minion-1.c.kubernetes-elk.internal/146.148.42.44     name=synth-logging-source    Running
influx-grafana                         influxdb                kubernetes/heapster_influxdb       kubernetes-minion-3.c.kubernetes-elk.internal/130.211.129.169   name=influxdb                Running
                                       grafana                 kubernetes/heapster_grafana                                                                                                     
                                       elasticsearch           dockerfile/elasticsearch                                                                                                        
heapster                               heapster                kubernetes/heapster                kubernetes-minion-2.c.kubernetes-elk.internal/146.148.41.87     name=heapster                Running
67cfcb1f-9764-11e4-898c-42010af03582   etcd                    quay.io/coreos/etcd:latest         kubernetes-minion-3.c.kubernetes-elk.internal/130.211.129.169   k8s-app=skydns               Running
                                       kube2sky                kubernetes/kube2sky:1.0                                                                                                         
                                       skydns                  kubernetes/skydns:2014-12-23-001                                                                                                
6ba20338-9764-11e4-898c-42010af03582   elasticsearch-logging   dockerfile/elasticsearch           kubernetes-minion-3.c.kubernetes-elk.internal/130.211.129.169   name=elasticsearch-logging   Running
../../../cluster/kubectl.sh get replicationControllers
Running: ../../../cluster/../cluster/gce/../../_output/dockerized/bin/linux/amd64/kubectl get replicationControllers
CONTROLLER                         CONTAINER(S)            IMAGE(S)                           SELECTOR                     REPLICAS
skydns                             etcd                    quay.io/coreos/etcd:latest         k8s-app=skydns               1
                                   kube2sky                kubernetes/kube2sky:1.0                                         
                                   skydns                  kubernetes/skydns:2014-12-23-001                                
elasticsearch-logging-controller   elasticsearch-logging   dockerfile/elasticsearch           name=elasticsearch-logging   1
kibana-logging-controller          kibana-logging          kubernetes/kibana                  name=kibana-logging          1
../../.../kubectl.sh get services
Running: ../../../cluster/../cluster/gce/../../_output/dockerized/bin/linux/amd64/kubectl get services
NAME                    LABELS                                    SELECTOR                     IP                  PORT
kubernetes-ro           component=apiserver,provider=kubernetes   <none>                       10.0.83.3           80
kubernetes              component=apiserver,provider=kubernetes   <none>                       10.0.79.4           443
influx-master           <none>                                    name=influxdb                10.0.232.223        8085
skydns                  k8s-app=skydns                            k8s-app=skydns               10.0.0.10           53
elasticsearch-logging   <none>                                    name=elasticsearch-logging   10.0.25.103         9200
kibana-logging          <none>                                    name=kibana-logging          10.0.208.114        5601

```
On the GCE provider you can also obtain the external IP addresses of the Elasticsearch and Kibana services:
```
$ make net
IPAddress: 130.211.120.118
IPProtocol: TCP
creationTimestamp: '2015-01-08T10:30:34.210-08:00'
id: '12815488049392139704'
kind: compute#forwardingRule
name: elasticsearch-logging
portRange: 9200-9200
region: https://www.googleapis.com/compute/v1/projects/kubernetes-elk/regions/us-central1
selfLink: https://www.googleapis.com/compute/v1/projects/kubernetes-elk/regions/us-central1/forwardingRules/elasticsearch-logging
target: https://www.googleapis.com/compute/v1/projects/kubernetes-elk/regions/us-central1/targetPools/elasticsearch-logging
gcloud compute forwarding-rules describe kibana-logging
IPAddress: 146.148.40.158
IPProtocol: TCP
creationTimestamp: '2015-01-08T10:31:05.715-08:00'
id: '2755171906970792849'
kind: compute#forwardingRule
name: kibana-logging
portRange: 5601-5601
region: https://www.googleapis.com/compute/v1/projects/kubernetes-elk/regions/us-central1
selfLink: https://www.googleapis.com/compute/v1/projects/kubernetes-elk/regions/us-central1/forwardingRules/kibana-logging
target: https://www.googleapis.com/compute/v1/projects/kubernetes-elk/regions/us-central1/targetPools/kibana-logging
```
