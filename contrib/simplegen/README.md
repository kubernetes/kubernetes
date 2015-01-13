# Simple configuration generation tool

`simplegen` is a command-line tool to expand a simple container
description into Kubernetes API objects, such as for consumption by
kubectl or other tools.

Currently targets only v1beta1.

### Usage
```
$ simplegen myservice.json
$ simplegen myservice.yaml
$ simplegen -
$ simplegen http://some.blog.site.com/k8s-example.yaml
```

### Schema
```
// Optional: Defaults to image base name if not specified
Name string `json:"name,omitempty"`
// Required.
Image string `json:"image"`
// Optional: Defaults to one
Replicas int `json:"replicas,omitempty"`
// Optional: Creates a service if specified: servicePort:containerPort
PortSpec string `json:"portSpec,omitempty"`
```

### Example
```
redismaster.yaml:
name: redismaster
image: dockerfile/redis
portSpec: 6379:6379

redisslave.yaml:
name: redisslave
image: brendanburns/redis-slave
replicas: 2
portSpec: 10001:6379
```
Output:
```
$ simplegen redismaster.yaml | cluster/kubectl.sh create -f -
$ simplegen redisslave.yaml | cluster/kubectl.sh create -f -
$ cluster/kubectl.sh get services
NAME                LABELS                      SELECTOR                                  IP                  PORT
kubernetes-ro                                   component=apiserver,provider=kubernetes   10.0.0.2            80
kubernetes                                      component=apiserver,provider=kubernetes   10.0.0.1            443
redismaster         simpleservice=redismaster   simpleservice=redismaster                 10.0.0.3            6379
redisslave          simpleservice=redisslave    simpleservice=redisslave                  10.0.0.4            10001
$ cluster/kubectl.sh get replicationcontrollers
NAME                IMAGE(S)                   SELECTOR                    REPLICAS
redismaster         dockerfile/redis           simpleservice=redismaster   1
redisslave          brendanburns/redis-slave   simpleservice=redisslave    2
$ cluster/kubectl.sh get pods
NAME                                   IMAGE(S)                   HOST                                                               LABELS                      STATUS
89adf546-6457-11e4-9f97-42010af0d824   dockerfile/redis           kubernetes-minion-3/146.148.79.186   simpleservice=redismaster   Running
93a555ac-6457-11e4-9f97-42010af0d824   brendanburns/redis-slave   kubernetes-minion-4/130.211.186.4    simpleservice=redisslave    Running
93a862d1-6457-11e4-9f97-42010af0d824   brendanburns/redis-slave   kubernetes-minion-1/130.211.117.14   simpleservice=redisslave    Running
```
