# Spark on GlusterFS example

This guide is an extension of the standard [Spark on Kubernetes Guide](https://github.com/GoogleCloudPlatform/kubernetes/tree/master/examples/spark) and describes how to run Spark on GlusterFS using the [Kubernetes Volume Plugin for GlusterFS](https://github.com/GoogleCloudPlatform/kubernetes/tree/master/examples/glusterfs)

The setup is the same in that you will setup a Spark Master Service in the same way you do with the standard Spark guide but you will deploy a modified Spark Master and a Modified Spark Worker ReplicationController, as they will be modified to use the GlusterFS volume plugin to mount a GlusterFS volume into the Spark Master and Spark Workers containers. Note that this example can be used as a guide for implementing any of the Kubernetes Volume Plugins with the Spark Example.

[There is also a video available that provides a walkthrough for how to set this solution up](https://youtu.be/xyIaoM0-gM0)

## Step Zero: Prerequisites

This example assumes that you have been able to successfully get the standard Spark Example working in Kubernetes and that you have a GlusterFS cluster that is accessible from your Kubernetes cluster. It is also recommended that you are familiar with the GlusterFS Volume Plugin and how to configure it. 

## Step One: Define the endpoints for your GlusterFS Cluster

Modify the `examples/spark/spark-gluster/glusterfs-endpoints.json` file to list the IP addresses of some of the servers in your GlusterFS cluster. The GlusterFS Volume Plugin uses these IP addresses to perform a Fuse Mount of the GlusterFS Volume into the Spark Worker Containers that are launched by the ReplicationController in the next section.

Register your endpoints by running the following command:

```shell
$ kubectl create -f examples/spark/spark-gluster/glusterfs-endpoints.json
```
## Step Two: Modify and Submit your Spark Master Pod

Modify the `examples/spark/spark-gluster/spark-master.json` file to reflect the GlusterFS Volume that you wish to use in the PATH parameter of the volumes subsection. 

Submit the Spark Master Pod 

```shell
$ kubectl create -f examples/spark/spark-gluster/spark-master.json
```
Verify that the Spark Master Pod deployed successfully.

```shell
$ kubectl get pods
```

Submit the Spark Master Service 
```shell
$ kubectl create -f examples/spark/spark-gluster/spark-master-service.json
```

Verify that the Spark Master Service deployed successfully.
```shell
$ kubectl get services
```

## Step Three: Start your Spark workers

Modify the `examples/spark/spark-gluster/spark-worker-rc.json` file to reflect the GlusterFS Volume that you wish to use in the PATH parameter of the Volumes subsection. 

Make sure that the replication factor for the pods is not greater than the amount of Kubernetes nodes available in your Kubernetes cluster.

Submit your Spark Worker ReplicationController by running the following command:

```shell
$ kubectl create -f examples/spark/spark-gluster/spark-worker-rc.json
```
Verify that the Spark Worker ReplicationController deployed its pods successfully.

```shell
$ kubectl get pods
```
Follow the steps from the standard example to verify the Spark Worker pods have registered successfully with the Spark Master.

## Step Four: Submit a Spark Job

All the Spark Workers and the Spark Master in your cluster have a mount to GlusterFS. This means that any of them can be used as the Spark Client to submit a job. For simplicity, lets use the Spark Master as an example.

Obtain the Host (Kubernetes Node) that the Spark Master container is running on
```shell
$ kubectl describe pod spark-master
```

Shell into the Kubernetes Node running the Spark Worker:

```shell
$ ssh root@worker-1
```

Identify the Container ID for the Spark Master:

```
$ docker ps
CONTAINER ID        IMAGE                                  COMMAND             CREATED             STATUS              PORTS                    NAMES
88a8531f9329        gcr.io/google_containers/spark-master:latest              "/start.sh"         4 minutes ago       Up 4 minutes                                 k8s_spark-master.af6b2d08_sgc-sfgj0_default_446a9f27-e8a2-11e4-ad4a-000c29151bdb_b2710ef5   
4a58e1f3489b        gcr.io/google_containers/pause:0.8.0   "/pause"            4 minutes ago       Up 4 minutes        0.0.0.0:8888->8088/tcp   k8s_POD.908f04ee_sgc-sfgj0_default_446a9f27-e8a2-11e4-ad4a-000c29151bdb_895cdc58 
```

Now we will shell into the Spark Master Container and run a Spark Job. In the example below, we are running the Spark Wordcount example and specifying the input and output directory at the location where GlusterFS is mounted in the Spark Master Container. This will submit the job to the Spark Master who will distribute the work to all the Spark Worker Containers. All the Spark Worker containers  will be able to access the data as they all have the same GlusterFS volume mounted at /mnt/glusterfs. The reason we are submitting the job from a Spark Worker and not an additional Spark Base container (as in the standard Spark Example) is due to the fact that the Spark instance submitting the job must be able to access the data. Only the Spark Master and Spark Worker containers have GlusterFS mounted.

The Spark Worker and Spark Master containers include a setup_client utility script that takes two parameters, the Service IP of the Spark Master and the port that it is running on. This must be to setup the container as a Spark client prior to submitting any Spark Jobs.

```
$ docker exec -it 88a8531f9329 sh
root@88a8531f9329:/# . /setup_client.sh 10.0.204.187 7077
root@88a8531f9329:/# pyspark
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

```shell
root@88a8531f9329:/# ls -l /mnt/glusterfs/output
```
