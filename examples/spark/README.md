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

Then, use the [`examples/spark/spark-master-service.json`](spar-master-service.json) file to
create a logical service endpoint that Spark workers can use to access
the Master pod.

```shell
$ kubectl create -f examples/spark/spark-master-service.json
```

### Check to see if Master is running and accessible

```shell
$ kubectl get pods
NAME                                           READY     STATUS    RESTARTS   AGE
[...]
spark-master                                   1/1       Running   0          25s

```

Check logs to see the status of the master.

```shell
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
[replication controller](../../docs/replication-controller.md) that manages the worker pods.

```shell
$ kubectl create -f examples/spark/spark-worker-controller.json
```

### Check to see if the workers are running

```shell
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
## Step Three: Do something with the cluster

Get the address and port of the Master service.

```shell
$ kubectl get service spark-master
NAME           LABELS              SELECTOR            IP(S)          PORT(S)
spark-master   name=spark-master   name=spark-master   10.0.204.187   7077/TCP
```

SSH to one of your cluster nodes. On GCE/GKE you can either use [Developers Console](https://console.developers.google.com)
(more details [here](https://cloud.google.com/compute/docs/ssh-in-browser))
or run  `gcloud compute ssh <name>` where the name can be taken from `kubectl get nodes`
(more details [here](https://cloud.google.com/compute/docs/gcloud-compute/#connecting)).

```
$ kubectl get nodes
NAME                     LABELS                                          STATUS
kubernetes-minion-5jvu   kubernetes.io/hostname=kubernetes-minion-5jvu   Ready
kubernetes-minion-6fbi   kubernetes.io/hostname=kubernetes-minion-6fbi   Ready
kubernetes-minion-8y2v   kubernetes.io/hostname=kubernetes-minion-8y2v   Ready
kubernetes-minion-h0tr   kubernetes.io/hostname=kubernetes-minion-h0tr   Ready

$ gcloud compute ssh kubernetes-minion-5jvu --zone=us-central1-b
Linux kubernetes-minion-5jvu 3.16.0-0.bpo.4-amd64 #1 SMP Debian 3.16.7-ckt9-3~deb8u1~bpo70+1 (2015-04-27) x86_64

=== GCE Kubernetes node setup complete ===

me@kubernetes-minion-5jvu:~$
```

Once logged in run spark-base image. Inside of the image there is a script
that sets up the environment based on the provided IP and port of the Master.

```
cluster-node $ sudo docker run -it gcr.io/google_containers/spark-base
root@f12a6fec45ce:/# . /setup_client.sh 10.0.204.187 7077
root@f12a6fec45ce:/# pyspark
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

You now have services, replication controllers, and pods for the Spark master and Spark workers.
You can take this example to the next step and start using the Apache Spark cluster 
you just created, see [Spark documentation](https://spark.apache.org/documentation.html) 
for more information.

## tl;dr

```kubectl create -f spark-master.json```

```kubectl create -f spark-master-service.json```

Make sure the Master Pod is running (use: ```kubectl get pods```).

```kubectl create -f spark-worker-controller.json```

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/spark/README.md?pixel)]()
