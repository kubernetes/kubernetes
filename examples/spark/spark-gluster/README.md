<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.3/examples/spark/spark-gluster/README.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Spark on GlusterFS example

This guide is an extension of the standard [Spark on Kubernetes Guide](../../../examples/spark/) and describes how to run Spark on GlusterFS using the [Kubernetes Volume Plugin for GlusterFS](../../../examples/volumes/glusterfs/)

The setup is the same in that you will setup a Spark Master Service in the same way you do with the standard Spark guide but you will deploy a modified Spark Master and a Modified Spark Worker ReplicationController, as they will be modified to use the GlusterFS volume plugin to mount a GlusterFS volume into the Spark Master and Spark Workers containers. Note that this example can be used as a guide for implementing any of the Kubernetes Volume Plugins with the Spark Example.

[There is also a video available that provides a walkthrough for how to set this solution up](https://youtu.be/xyIaoM0-gM0)

## Step Zero: Prerequisites

This example assumes that you have been able to successfully get the standard Spark Example working in Kubernetes and that you have a GlusterFS cluster that is accessible from your Kubernetes cluster. It is also recommended that you are familiar with the GlusterFS Volume Plugin and how to configure it.

## Step One: Define the endpoints for your GlusterFS Cluster

Modify the `examples/spark/spark-gluster/glusterfs-endpoints.yaml` file to list the IP addresses of some of the servers in your GlusterFS cluster. The GlusterFS Volume Plugin uses these IP addresses to perform a Fuse Mount of the GlusterFS Volume into the Spark Worker Containers that are launched by the ReplicationController in the next section.

Register your endpoints by running the following command:

```console
$ kubectl create -f examples/spark/spark-gluster/glusterfs-endpoints.yaml
```

## Step Two: Modify and Submit your Spark Master ReplicationController

Modify the `examples/spark/spark-gluster/spark-master-controller.yaml` file to reflect the GlusterFS Volume that you wish to use in the PATH parameter of the volumes subsection.

Submit the Spark Master Pod

```console
$ kubectl create -f examples/spark/spark-gluster/spark-master-controller.yaml
```

Verify that the Spark Master Pod deployed successfully.

```console
$ kubectl get pods
```

Submit the Spark Master Service

```console
$ kubectl create -f examples/spark/spark-gluster/spark-master-service.yaml
```

Verify that the Spark Master Service deployed successfully.

```console
$ kubectl get services
```

## Step Three: Start your Spark workers

Modify the `examples/spark/spark-gluster/spark-worker-controller.yaml` file to reflect the GlusterFS Volume that you wish to use in the PATH parameter of the Volumes subsection.

Make sure that the replication factor for the pods is not greater than the amount of Kubernetes nodes available in your Kubernetes cluster.

Submit your Spark Worker ReplicationController by running the following command:

```console
$ kubectl create -f examples/spark/spark-gluster/spark-worker-controller.yaml
```

Verify that the Spark Worker ReplicationController deployed its pods successfully.

```console
$ kubectl get pods
```

Follow the steps from the standard example to verify the Spark Worker pods have registered successfully with the Spark Master.

## Step Four: Submit a Spark Job

All the Spark Workers and the Spark Master in your cluster have a mount to GlusterFS. This means that any of them can be used as the Spark Client to submit a job. For simplicity, lets use the Spark Master as an example.


The Spark Worker and Spark Master containers include a setup_client utility script that takes two parameters, the Service IP of the Spark Master and the port that it is running on. This must be to setup the container as a Spark client prior to submitting any Spark Jobs.

Obtain the Service IP (listed as IP:) and Full Pod Name by running

```console
$ kubectl describe pod spark-master-controller
```

Now we will shell into the Spark Master Container and run a Spark Job. In the example below, we are running the Spark Wordcount example and specifying the input and output directory at the location where GlusterFS is mounted in the Spark Master Container. This will submit the job to the Spark Master who will distribute the work to all the Spark Worker Containers.

All the Spark Worker containers  will be able to access the data as they all have the same GlusterFS volume mounted at /mnt/glusterfs. The reason we are submitting the job from a Spark Worker and not an additional Spark Base container (as in the standard Spark Example) is due to the fact that the Spark instance submitting the job must be able to access the data. Only the Spark Master and Spark Worker containers have GlusterFS mounted.

The Spark Worker and Spark Master containers include a setup_client utility script that takes two parameters, the Service IP of the Spark Master and the port that it is running on. This must be done to setup the container as a Spark client prior to submitting any Spark Jobs.

Shell into the Master Spark Node (spark-master-controller) by running

```console
kubectl exec spark-master-controller-<ID> -i -t -- bash -i

root@spark-master-controller-c1sqd:/# . /setup_client.sh <Service IP> 7077
root@spark-master-controller-c1sqd:/# pyspark

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
>>> file = sc.textFile("/mnt/glusterfs/somefile.txt")
>>> counts = file.flatMap(lambda line: line.split(" ")).map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b)
>>> counts.saveAsTextFile("/mnt/glusterfs/output")
```

While still in the container, you can see the output of your Spark Job in the Distributed File System by running the following:

```console
root@spark-master-controller-c1sqd:/# ls -l /mnt/glusterfs/output
```

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/spark/spark-gluster/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
