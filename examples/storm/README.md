# Storm example

Following this example, you will create a functional [Apache
Storm](http://storm.apache.org/) cluster using Kubernetes and
[Docker](http://docker.io).

You will setup an [Apache ZooKeeper](http://zookeeper.apache.org/)
service, a Storm master service (a.k.a. Nimbus server), and a set of
Storm workers (a.k.a. supervisors).

## tl;dr

```kubectl create -f zookeeper.json```

```kubectl create -f zookeeper-service.json```

Make sure the ZooKeeper Pod is running (use: ```kubectl get pods```).

```kubectl create -f storm-nimbus.json```

```kubectl create -f storm-nimbus-service.json```

Make sure the Nimbus Pod is running.

```kubectl create -f storm-worker-controller.json```
