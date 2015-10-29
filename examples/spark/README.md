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

The Master [service](../../docs/user-guide/services.md) is the master service
for a Spark cluster.

Use the
[`examples/spark/spark-master-controller.yaml`](spark-master-controller.yaml)
file to create a
[replication controller](../../docs/user-guide/replication-controller.md)
running the Spark Master service.

```console
$ kubectl create -f examples/spark/spark-master-controller.yaml
replicationcontrollers/spark-master-controller
```

Then, use the
[`examples/spark/spark-master-service.yaml`](spark-master-service.yaml) file to
create a logical service endpoint that Spark workers can use to access the
Master pod.

```console
$ kubectl create -f examples/spark/spark-master-service.yaml
services/spark-master
```

Optionally, you can create a service for the Spark Master WebUI at this point as
well. If you are running on a cloud provider that supports it, this will create
an external load balancer and open a firewall to the Spark Master WebUI on the
cluster. **Note:** With the existing configuration, there is **ABSOLUTELY NO**
authentication on this WebUI. With slightly more work, it would be
straightforward to put an `nginx` proxy in front to password protect it.

```console
$ kubectl create -f examples/spark/spark-webui.yaml
services/spark-webui
```

### Check to see if Master is running and accessible

```console
$ kubectl get pods
NAME                            READY     STATUS    RESTARTS   AGE
spark-master-controller-5u0q5   1/1       Running   0          8m
```

Check logs to see the status of the master. (Use the pod retrieved from the previous output.)

```sh
$ kubectl logs spark-master-controller-5u0q5
starting org.apache.spark.deploy.master.Master, logging to /opt/spark-1.5.1-bin-hadoop2.6/sbin/../logs/spark--org.apache.spark.deploy.master.Master-1-spark-master-controller-g0oao.out
Spark Command: /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java -cp /opt/spark-1.5.1-bin-hadoop2.6/sbin/../conf/:/opt/spark-1.5.1-bin-hadoop2.6/lib/spark-assembly-1.5.1-hadoop2.6.0.jar:/opt/spark-1.5.1-bin-hadoop2.6/lib/datanucleus-rdbms-3.2.9.jar:/opt/spark-1.5.1-bin-hadoop2.6/lib/datanucleus-core-3.2.10.jar:/opt/spark-1.5.1-bin-hadoop2.6/lib/datanucleus-api-jdo-3.2.6.jar -Xms1g -Xmx1g org.apache.spark.deploy.master.Master --ip spark-master --port 7077 --webui-port 8080
========================================
15/10/27 21:25:05 INFO Master: Registered signal handlers for [TERM, HUP, INT]
15/10/27 21:25:05 INFO SecurityManager: Changing view acls to: root
15/10/27 21:25:05 INFO SecurityManager: Changing modify acls to: root
15/10/27 21:25:05 INFO SecurityManager: SecurityManager: authentication disabled; ui acls disabled; users with view permissions: Set(root); users with modify permissions: Set(root)
15/10/27 21:25:06 INFO Slf4jLogger: Slf4jLogger started
15/10/27 21:25:06 INFO Remoting: Starting remoting
15/10/27 21:25:06 INFO Remoting: Remoting started; listening on addresses :[akka.tcp://sparkMaster@spark-master:7077]
15/10/27 21:25:06 INFO Utils: Successfully started service 'sparkMaster' on port 7077.
15/10/27 21:25:07 INFO Master: Starting Spark master at spark://spark-master:7077
15/10/27 21:25:07 INFO Master: Running Spark version 1.5.1
15/10/27 21:25:07 INFO Utils: Successfully started service 'MasterUI' on port 8080.
15/10/27 21:25:07 INFO MasterWebUI: Started MasterWebUI at http://spark-master:8080
15/10/27 21:25:07 INFO Utils: Successfully started service on port 6066.
15/10/27 21:25:07 INFO StandaloneRestServer: Started REST server for submitting applications on port 6066
15/10/27 21:25:07 INFO Master: I have been elected leader! New state: ALIVE
```

If you created the Spark WebUI and waited sufficient time for the load balancer
to be create, the `spark-webui` service should look something like this:

```console
$ kubectl describe services/spark-webui
Name:                   spark-webui
Namespace:              default
Labels:                 <none>
Selector:               component=spark-master
Type:                   LoadBalancer
IP:                     10.0.152.249
LoadBalancer Ingress:   104.197.147.190
Port:                   <unnamed>       8080/TCP
NodePort:               <unnamed>       31141/TCP
Endpoints:              10.244.1.12:8080
Session Affinity:       None
Events: [...]
```

You should now be able to visit `http://104.197.147.190:8080` and see the Spark
Master UI. *Note:* After workers connect, this UI has links to worker Web
UIs. The worker UI links do not work (the links attempt to connect to cluster
IPs).

## Step Two: Start your Spark workers

The Spark workers do the heavy lifting in a Spark cluster. They
provide execution resources and data cache capabilities for your
program.

The Spark workers need the Master service to be running.

Use the [`examples/spark/spark-worker-controller.yaml`](spark-worker-controller.yaml) file to create a
[replication controller](../../docs/user-guide/replication-controller.md) that manages the worker pods.

```console
$ kubectl create -f examples/spark/spark-worker-controller.yaml
```

### Check to see if the workers are running

If you launched the Spark WebUI, your workers should just appear in the UI when
they're ready. (It may take a little bit to pull the images and launch the
pods.) You can also interrogate the status in the following way:

```console
$ kubectl get pods
NAME                            READY     STATUS    RESTARTS   AGE
spark-master-controller-5u0q5   1/1       Running   0          25m
spark-worker-controller-e8otp   1/1       Running   0          6m
spark-worker-controller-fiivl   1/1       Running   0          6m
spark-worker-controller-ytc7o   1/1       Running   0          6m

$ kubectl logs spark-master-controller-5u0q5
[...]
15/10/26 18:20:14 INFO Master: Registering worker 10.244.1.13:53567 with 2 cores, 6.3 GB RAM
15/10/26 18:20:14 INFO Master: Registering worker 10.244.2.7:46195 with 2 cores, 6.3 GB RAM
15/10/26 18:20:14 INFO Master: Registering worker 10.244.3.8:39926 with 2 cores, 6.3 GB RAM
```

## Step Three: Start your Spark driver to launch jobs on your Spark cluster

The Spark driver is used to launch jobs into Spark cluster. You can read more about it in
[Spark architecture](https://spark.apache.org/docs/latest/cluster-overview.html).

```console
$ kubectl create -f examples/spark/spark-driver-controller.yaml
replicationcontrollers/spark-driver-controller
```

The Spark driver needs the Master service to be running.

### Check to see if the driver is running

```console
$ kubectl get pods -lcomponent=spark-driver
NAME                            READY     STATUS    RESTARTS   AGE
spark-driver-controller-vwb9c   1/1       Running   0          1m
```

## Step Four: Do something with the cluster

Use the kubectl exec to connect to Spark driver and run a pipeline.

```console
$ kubectl exec spark-driver-controller-vwb9c -it pyspark
Python 2.7.9 (default, Mar  1 2015, 12:57:24)
[GCC 4.9.2] on linux2
Type "help", "copyright", "credits" or "license" for more information.
Welcome to
      ____              __
     / __/__  ___ _____/ /__
    _\ \/ _ \/ _ `/ __/  '_/
   /__ / .__/\_,_/_/ /_/\_\   version 1.5.1
      /_/

Using Python version 2.7.9 (default, Mar  1 2015 12:57:24)
SparkContext available as sc, HiveContext available as sqlContext.
>>> sc.textFile("gs://dataflow-samples/shakespeare/*").map(lambda s: len(s.split())).sum()
939193
```

Congratulations, you just counted all of the words in all of the plays of
Shakespeare.

## Result

You now have services and replication controllers for the Spark master, Spark
workers and Spark driver.  You can take this example to the next step and start
using the Apache Spark cluster you just created, see
[Spark documentation](https://spark.apache.org/documentation.html) for more
information.

## tl;dr

```console
kubectl create -f examples/spark/spark-master-controller.yaml
kubectl create -f examples/spark/spark-master-service.yaml
kubectl create -f examples/spark/spark-webui.yaml
kubectl create -f examples/spark/spark-worker-controller.yaml
kubectl create -f examples/spark/spark-driver-controller.yaml
```

After it's setup:

```console
kubectl get pods # Make sure everything is running
kubectl get services spark-webui # Get the IP of the Spark WebUI
kubectl get pods -lcomponent=spark-driver # Get the driver pod to interact with.
```

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/spark/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
