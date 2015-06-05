# Spark example

Following this example, you will create a functional [Apache
Spark](http://spark.apache.org/) cluster using Kubernetes and
[Docker](http://docker.io).

You will setup a Spark master service and a set of
Spark workers using Spark's [standalone mode](http://spark.apache.org/docs/latest/spark-standalone.html).

For the impatient expert, jump straight to the [tl;dr](#tldr)
section.

### Sources

Source is freely available at:
* Docker image - https://github.com/mattf/docker-spark
* Docker Trusted Build - https://registry.hub.docker.com/search?q=mattf/spark

## Step Zero: Prerequisites

This example assumes you have a Kubernetes cluster installed and
running, and that you have installed the ```kubectl``` command line
tool somewhere in your path. Please see the [getting
started](../../docs/getting-started-guides) for installation
instructions for your platform.

## Step One: Start your Master service

The Master [service](../../docs/services.md) is the master (or head) service for a Spark
cluster.

Use the [`examples/spark/spark-master.json`](spark-master.json) file to create a [pod](../../docs/pods.md) running
the Master service.

```shell
$ kubectl create -f examples/spark/spark-master.json
```

Then, use the `examples/spark/spark-master-service.json` file to
create a logical service endpoint that Spark workers can use to access
the Master pod.

```shell
$ kubectl create -f examples/spark/spark-master-service.json
```

Ensure that the Master service is running and functional.

### Check to see if Master is running and accessible

```shell
$ kubectl get pods,services
POD                             IP                  CONTAINER(S)        IMAGE(S)             HOST                          LABELS                                STATUS
spark-master                    192.168.90.14       spark-master        mattf/spark-master   172.18.145.8/172.18.145.8     name=spark-master                     Running
NAME                LABELS                                    SELECTOR            IP                  PORT
kubernetes          component=apiserver,provider=kubernetes   <none>              10.254.0.2          443
spark-master        name=spark-master                         name=spark-master   10.254.125.166      7077
```

Connect to http://192.168.90.14:8080 to see the status of the master.

```shell
$ links -dump 192.168.90.14:8080
  [IMG] 1.2.1 Spark Master at spark://spark-master:7077

     * URL: spark://spark-master:7077
     * Workers: 0
     * Cores: 0 Total, 0 Used
     * Memory: 0.0 B Total, 0.0 B Used
     * Applications: 0 Running, 0 Completed
     * Drivers: 0 Running, 0 Completed
     * Status: ALIVE
...
```

(Pull requests welcome for an alternative that uses the service IP and
port)

## Step Two: Start your Spark workers

The Spark workers do the heavy lifting in a Spark cluster. They
provide execution resources and data cache capabilities for your
program.

The Spark workers need the Master service to be running.

Use the [`examples/spark/spark-worker-controller.json`](spark-worker-controller.json) file to create a
[replication controller](../../docs/replication-controller.md) that manages the worker pods.

```shell
$ kubectl create -f examples/spark/spark-worker-controller.json
```

### Check to see if the workers are running

```shell
$ links -dump 192.168.90.14:8080
  [IMG] 1.2.1 Spark Master at spark://spark-master:7077

     * URL: spark://spark-master:7077
     * Workers: 3
     * Cores: 12 Total, 0 Used
     * Memory: 20.4 GB Total, 0.0 B Used
     * Applications: 0 Running, 0 Completed
     * Drivers: 0 Running, 0 Completed
     * Status: ALIVE

    Workers

Id                                        Address             State Cores Memory
                                                                    4 (0  6.8 GB
worker-20150318151745-192.168.75.14-46422 192.168.75.14:46422 ALIVE Used) (0.0 B
                                                                          Used)
                                                                    4 (0  6.8 GB
worker-20150318151746-192.168.35.17-53654 192.168.35.17:53654 ALIVE Used) (0.0 B
                                                                          Used)
                                                                    4 (0  6.8 GB
worker-20150318151746-192.168.90.17-50490 192.168.90.17:50490 ALIVE Used) (0.0 B
                                                                          Used)
...
```

(Pull requests welcome for an alternative that uses the service IP and
port)

## Step Three: Do something with the cluster

```shell
$ kubectl get pods,services
POD                             IP                  CONTAINER(S)        IMAGE(S)             HOST                          LABELS                                STATUS
spark-master                    192.168.90.14       spark-master        mattf/spark-master   172.18.145.8/172.18.145.8     name=spark-master                     Running
spark-worker-controller-51wgg   192.168.75.14       spark-worker        mattf/spark-worker   172.18.145.9/172.18.145.9     name=spark-worker,uses=spark-master   Running
spark-worker-controller-5v48c   192.168.90.17       spark-worker        mattf/spark-worker   172.18.145.8/172.18.145.8     name=spark-worker,uses=spark-master   Running
spark-worker-controller-ehq23   192.168.35.17       spark-worker        mattf/spark-worker   172.18.145.12/172.18.145.12   name=spark-worker,uses=spark-master   Running
NAME                LABELS                                    SELECTOR            IP                  PORT
kubernetes          component=apiserver,provider=kubernetes   <none>              10.254.0.2          443
spark-master        name=spark-master                         name=spark-master   10.254.125.166      7077

$ sudo docker run -it mattf/spark-base sh

sh-4.2# echo "10.254.125.166 spark-master" >> /etc/hosts

sh-4.2# export SPARK_LOCAL_HOSTNAME=$(hostname -i)

sh-4.2# MASTER=spark://spark-master:7077 pyspark
Python 2.7.5 (default, Jun 17 2014, 18:11:42)
[GCC 4.8.2 20140120 (Red Hat 4.8.2-16)] on linux2
Type "help", "copyright", "credits" or "license" for more information.
Welcome to
      ____              __
     / __/__  ___ _____/ /__
    _\ \/ _ \/ _ `/ __/  '_/
   /__ / .__/\_,_/_/ /_/\_\   version 1.2.1
      /_/

Using Python version 2.7.5 (default, Jun 17 2014 18:11:42)
SparkContext available as sc.
>>> import socket, resource
>>> sc.parallelize(range(1000)).map(lambda x: (socket.gethostname(), resource.getrlimit(resource.RLIMIT_NOFILE))).distinct().collect()
[('spark-worker-controller-ehq23', (1048576, 1048576)), ('spark-worker-controller-5v48c', (1048576, 1048576)), ('spark-worker-controller-51wgg', (1048576, 1048576))]
```

## tl;dr

```kubectl create -f spark-master.json```

```kubectl create -f spark-master-service.json```

Make sure the Master Pod is running (use: ```kubectl get pods```).

```kubectl create -f spark-worker-controller.json```


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/spark/README.md?pixel)]()
