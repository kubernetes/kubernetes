<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<strong>
The latest 1.0.x release of this document can be found
[here](http://releases.k8s.io/release-1.0/examples/spark/README.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Spark example

Following this example, you will create a functional [Apache
Spark](http://spark.apache.org/) cluster using Kubernetes and
[Docker](http://docker.io).

You will setup a Spark master service and a set of
Spark workers using Spark's [standalone mode](http://spark.apache.org/docs/latest/spark-standalone.html).

For the impatient expert, jump straight to the [tl;dr](#tldr)
section.

### Sources

The Docker images are heavily based on https://github.com/mattf/docker-spark

## Step Zero: Prerequisites

This example assumes you have a Kubernetes cluster installed and
running, and that you have installed the ```kubectl``` command line
tool somewhere in your path. Please see the [getting
started](../../docs/getting-started-guides/) for installation
instructions for your platform.

## Step One: Start your Master service

The Master [service](../../docs/user-guide/services.md) is the master (or head) service for a Spark
cluster.

Use the [`examples/spark/spark-master.json`](spark-master.json) file to create a [pod](../../docs/user-guide/pods.md) running
the Master service.

```sh
$ kubectl create -f examples/spark/spark-master.json
```

Then, use the [`examples/spark/spark-master-service.json`](spark-master-service.json) file to
create a logical service endpoint that Spark workers can use to access
the Master pod.

```sh
$ kubectl create -f examples/spark/spark-master-service.json
```

### Check to see if Master is running and accessible

```sh
$ kubectl get pods
NAME                                           READY     STATUS    RESTARTS   AGE
[...]
spark-master                                   1/1       Running   0          25s

```

Check logs to see the status of the master.

```sh
$ kubectl logs spark-master

starting org.apache.spark.deploy.master.Master, logging to /opt/spark-1.4.0-bin-hadoop2.6/sbin/../logs/spark--org.apache.spark.deploy.master.Master-1-spark-master.out
Spark Command: /usr/lib/jvm/java-7-openjdk-amd64/jre/bin/java -cp /opt/spark-1.4.0-bin-hadoop2.6/sbin/../conf/:/opt/spark-1.4.0-bin-hadoop2.6/lib/spark-assembly-1.4.0-hadoop2.6.0.jar:/opt/spark-1.4.0-bin-hadoop2.6/lib/datanucleus-api-jdo-3.2.6.jar:/opt/spark-1.4.0-bin-hadoop2.6/lib/datanucleus-rdbms-3.2.9.jar:/opt/spark-1.4.0-bin-hadoop2.6/lib/datanucleus-core-3.2.10.jar -Xms512m -Xmx512m -XX:MaxPermSize=128m org.apache.spark.deploy.master.Master --ip spark-master --port 7077 --webui-port 8080
========================================
15/06/26 14:01:49 INFO Master: Registered signal handlers for [TERM, HUP, INT]
15/06/26 14:01:50 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
15/06/26 14:01:51 INFO SecurityManager: Changing view acls to: root
15/06/26 14:01:51 INFO SecurityManager: Changing modify acls to: root
15/06/26 14:01:51 INFO SecurityManager: SecurityManager: authentication disabled; ui acls disabled; users with view permissions: Set(root); users with modify permissions: Set(root)
15/06/26 14:01:51 INFO Slf4jLogger: Slf4jLogger started
15/06/26 14:01:51 INFO Remoting: Starting remoting
15/06/26 14:01:52 INFO Remoting: Remoting started; listening on addresses :[akka.tcp://sparkMaster@spark-master:7077]
15/06/26 14:01:52 INFO Utils: Successfully started service 'sparkMaster' on port 7077.
15/06/26 14:01:52 INFO Utils: Successfully started service on port 6066.
15/06/26 14:01:52 INFO StandaloneRestServer: Started REST server for submitting applications on port 6066
15/06/26 14:01:52 INFO Master: Starting Spark master at spark://spark-master:7077
15/06/26 14:01:52 INFO Master: Running Spark version 1.4.0
15/06/26 14:01:52 INFO Utils: Successfully started service 'MasterUI' on port 8080.
15/06/26 14:01:52 INFO MasterWebUI: Started MasterWebUI at http://10.244.2.34:8080
15/06/26 14:01:53 INFO Master: I have been elected leader! New state: ALIVE
```

## Step Two: Start your Spark workers

The Spark workers do the heavy lifting in a Spark cluster. They
provide execution resources and data cache capabilities for your
program.

The Spark workers need the Master service to be running.

Use the [`examples/spark/spark-worker-controller.json`](spark-worker-controller.json) file to create a
[replication controller](../../docs/user-guide/replication-controller.md) that manages the worker pods.

```sh
$ kubectl create -f examples/spark/spark-worker-controller.json
```

### Check to see if the workers are running

```sh
$ kubectl get pods
NAME                                            READY     STATUS    RESTARTS   AGE
[...]
spark-master                                    1/1       Running   0          14m
spark-worker-controller-hifwi                   1/1       Running   0          33s
spark-worker-controller-u40r2                   1/1       Running   0          33s
spark-worker-controller-vpgyg                   1/1       Running   0          33s

$ kubectl logs spark-master
[...]
15/06/26 14:15:43 INFO Master: Registering worker 10.244.2.35:46199 with 1 cores, 2.6 GB RAM
15/06/26 14:15:55 INFO Master: Registering worker 10.244.1.15:44839 with 1 cores, 2.6 GB RAM
15/06/26 14:15:55 INFO Master: Registering worker 10.244.0.19:60970 with 1 cores, 2.6 GB RAM
```

## Step Three: Start your Spark driver to launch jobs on your Spark cluster

The Spark driver is used to launch jobs into Spark cluster. You can read more about it in
[Spark architecture](http://spark.apache.org/docs/latest/cluster-overview.html).

```shell
$ kubectl create -f examples/spark/spark-driver.json
```

The Spark driver needs the Master service to be running.

### Check to see if the driver is running

```shell
$ kubectl get pods
NAME                                           READY     REASON    RESTARTS   AGE
[...]
spark-master                                    1/1       Running   0          14m
spark-driver                                    1/1       Running   0          10m
```

## Step Four: Do something with the cluster

Use the kubectl exec to connect to Spark driver

```
$ kubectl exec spark-driver -it bash
root@spark-driver:/#
root@spark-driver:/# pyspark
Python 2.7.9 (default, Mar  1 2015, 12:57:24)
[GCC 4.9.2] on linux2
Type "help", "copyright", "credits" or "license" for more information.
15/06/26 14:25:28 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
Welcome to
      ____              __
     / __/__  ___ _____/ /__
    _\ \/ _ \/ _ `/ __/  '_/
   /__ / .__/\_,_/_/ /_/\_\   version 1.4.0
      /_/
Using Python version 2.7.9 (default, Mar  1 2015 12:57:24)
SparkContext available as sc, HiveContext available as sqlContext.
>>> import socket
>>> sc.parallelize(range(1000)).map(lambda x:socket.gethostname()).distinct().collect()
['spark-worker-controller-u40r2', 'spark-worker-controller-hifwi', 'spark-worker-controller-vpgyg']
```

## Result

You now have services, replication controllers, and pods for the Spark master , Spark driver and Spark workers.
You can take this example to the next step and start using the Apache Spark cluster
you just created, see [Spark documentation](https://spark.apache.org/documentation.html)
for more information.

## tl;dr

```kubectl create -f spark-master.json```

```kubectl create -f spark-master-service.json```

Make sure the Master Pod is running (use: ```kubectl get pods```).

```kubectl create -f spark-worker-controller.json```

```kubectl create -f spark-driver.json```


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/spark/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
