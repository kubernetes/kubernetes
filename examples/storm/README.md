# Storm example

Following this example, you will create a functional [Apache
Storm](http://storm.apache.org/) cluster using Kubernetes and
[Docker](http://docker.io).

You will setup an [Apache ZooKeeper](http://zookeeper.apache.org/)
service, a Storm master service (a.k.a. Nimbus server), and a set of
Storm workers (a.k.a. supervisors).

## Step Zero: Prerequisites

This example assumes you have a Kubernetes cluster installed and
running, and that you have installed the ```kubectl``` command line
tool somewhere in your path. Please see the [getting
started](../../docs/getting-started-guides) for installation
instructions for your platform.

## Step One: Start your ZooKeeper service

ZooKeeper is a distributed coordination service that Storm uses as a
bootstrap and for state storage.

Use the `examples/storm/zookeeper.json` file to create a pod running
the ZooKeeper service.

```shell
$ kubectl create -f examples/storm/zookeeper.json
```

Then, use the `examples/storm/zookeeper-service.json` file to create a
logical service endpoint that Storm can use to access the ZooKeeper
pod.

```shell
$ kubectl create -f examples/storm/zookeeper-service.json
```

You should make sure the ZooKeeper pod is Running and accessible
before proceeding.

### Check to see if ZooKeeper is running

```shell
$ kubectl get pods
POD                 IP                  CONTAINER(S)        IMAGE(S)             HOST                          LABELS                      STATUS
zookeeper           192.168.86.4        zookeeper           mattf/zookeeper      172.18.145.8/172.18.145.8     name=zookeeper              Running
```

### Check to see if ZooKeeper is accessible

```shell
$ kubectl get services
NAME                LABELS                                    SELECTOR            IP                  PORT
kubernetes          component=apiserver,provider=kubernetes   <none>              10.254.0.2          443
kubernetes-ro       component=apiserver,provider=kubernetes   <none>              10.254.0.1          80
zookeeper           name=zookeeper                            name=zookeeper      10.254.139.141      2181

$ echo ruok | nc 10.254.139.141 2181; echo
imok
```

## Step Two: Start your Nimbus service

The Nimbus service is the master (or head) service for a Storm
cluster. It depends on a functional ZooKeeper service.

Use the `examples/storm/storm-nimbus.json` file to create a pod running
the Nimbus service.

```shell
$ kubectl create -f examples/storm/storm-nimbus.json
```

Then, use the `examples/storm/storm-nimbus-service.json` file to
create a logical service endpoint that Storm workers can use to access
the Nimbus pod.

```shell
$ kubectl create -f examples/storm/storm-nimbus-service.json
```

Ensure that the Nimbus service is running and functional.

### Check to see if Nimbus is running and accessible

```shell
$ kubectl get services
NAME                LABELS                                    SELECTOR            IP                  PORT
kubernetes          component=apiserver,provider=kubernetes   <none>              10.254.0.2          443
kubernetes-ro       component=apiserver,provider=kubernetes   <none>              10.254.0.1          80
nimbus              name=nimbus                               name=nimbus         10.254.115.208      6627
zookeeper           name=zookeeper                            name=zookeeper      10.254.139.141      2181

$ sudo docker run -it -w /opt/apache-storm mattf/storm-base sh -c '/configure.sh 10.254.139.141 10.254.115.208; ./bin/storm list'
...
No topologies running.
```

## tl;dr

```kubectl create -f zookeeper.json```

```kubectl create -f zookeeper-service.json```

Make sure the ZooKeeper Pod is running (use: ```kubectl get pods```).

```kubectl create -f storm-nimbus.json```

```kubectl create -f storm-nimbus-service.json```

Make sure the Nimbus Pod is running.

```kubectl create -f storm-worker-controller.json```
