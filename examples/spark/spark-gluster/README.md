# Spark on GlusterFS example

This guide is an extension of the standard [Spark on Kubernetes Guide](https://github.com/GoogleCloudPlatform/kubernetes/tree/master/examples/spark) and describes how to run Spark on GlusterFS using the [Kubernetes Volume Plugin for GlusterFS](https://github.com/GoogleCloudPlatform/kubernetes/tree/master/examples/glusterfs)

Essentially, the setup is the same in that you will setup a Spark Master Service and Spark Master pod the same way you do with the standard Spark guide but you will deploy a different ReplicationController and submit the Spark Jobs a slightly different way. 

## Step Zero: Prerequisites

This example assumes that you have been able to successfully get the standard Spark Example working in Kubernetes and that you have a GlusterFS cluster that is accessible from your Kubernetes cluster. It is also recommended that you are familiar with the GlusterFS Volume Plugin and how to configure it. 

## Step One: Start your Spark Master Service and Pod

Follow the standard Spark Example to deploy the Spark Master Pod and Spark Master Service. Follow the instructions to verify that the Master Service is running and functional

## Step Two: Define the endpoints for your GlusterFS Cluster

Modify the `examples/spark/spark-gluster/glusterfs-endpoints.json` file to list the IP addresses of some of the servers in your GlusterFS cluster. The GlusterFS Volume Plugin uses these IP addresses to perform a Fuse Mount of the GlusterFS Volume into the Spark Worker Containers that are launched by the ReplicationController in the next section.

Register your endpoints by running the following command:

```shell
$ kubectl create -f examples/spark/spark-gluster/glusterfs-endpoints.json
```

## Step Three: Start your Spark workers

This example uses a different ReplicationController to the standard example due to the fact that it must be modified to specify the usage of the Volume Plugin for GlusterFS.

Modify the `examples/spark/spark-gluster/spark-gluster-controller.json` file to reflect your endpoints and your GlusterFS Volume. In this example, it is set to "MyVolume".  Make sure that the replication factor for the pods is not greater than the amount of Kubernetes nodes available in your Kubernetes cluster.

Submit your ReplicationController by running the following command:

```shell
$ kubectl create -f examples/spark/spark-gluster/spark-gluster-controller.json
```

Follow the steps from the standard example to verify the Spark Worker pods have registered successfully with the Spark Master.

## Step Four: Submit a Spark Job

Identify a Kubernetes Node in your cluster that is running a Spark Worker, by running the following command:

```shell
$ kubectl get pods
POD                             IP                  CONTAINER(S)        IMAGE(S)             HOST                          LABELS                                STATUS
spark-master                    192.168.90.14       spark-master        mattf/spark-master   172.18.145.8/172.18.145.8     name=spark-master                     Running
spark-worker-controller-51wgg   192.168.75.14       spark-worker        mattf/spark-worker   172.18.145.9/172.18.145.9     name=spark-worker,uses=spark-master   Running
spark-worker-controller-5v48c   192.168.90.17       spark-worker        mattf/spark-worker   172.18.145.8/172.18.145.8     name=spark-worker,uses=spark-master   Running
spark-worker-controller-ehq23   192.168.35.17       spark-worker        mattf/spark-worker   172.18.145.12/172.18.145.12   name=spark-worker,uses=spark-master   Running
```

Shell into the Kubernetes Node running the Spark Worker:

```
$ ssh root@192.168.35.17
```

Identify the Container ID for the Spark Worker:

```
$ docker ps
CONTAINER ID        IMAGE                                  COMMAND             CREATED             STATUS              PORTS                    NAMES
88a8531f9329        mattf/spark-worker:latest              "/start.sh"         4 minutes ago       Up 4 minutes                                 k8s_spark-worker.af6b2d08_sgc-sfgj0_default_446a9f27-e8a2-11e4-ad4a-000c29151bdb_b2710ef5   
4a58e1f3489b        gcr.io/google_containers/pause:0.8.0   "/pause"            4 minutes ago       Up 4 minutes        0.0.0.0:8888->8088/tcp   k8s_POD.908f04ee_sgc-sfgj0_default_446a9f27-e8a2-11e4-ad4a-000c29151bdb_895cdc58 
```

Now we will shell into the Spark Worker Container and run a Spark Job. In the example below, we are running the Spark Wordcount example and specifying the input and output directory at the location where GlusterFS is mounted in the Spark Worker Container. This will submit the job to the Spark Master who will distribute the work to all the Spark Worker Containers. All the Spark Worker containers  will be able to access the data as they all have the same GlusterFS volume mounted at /mnt/glusterfs. The reason we are submitting the job from a Spark Worker and not an additional Spark Base container (as in the standard Spark Example) is due to the fact that the Spark instance submitting the job must be able to access the data. Only the Spark Worker containers have GlusterFS mounted. One could also configure the Spark Master Pod to use the GlusterFS Volume Plugin and mount the GlusterFS Volume into the Spark Master container and that would allow you to also submit jobs using the Spark Master container.

```
$ docker exec -it 88a8531f9329 sh
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
>>> file = sc.textFile("/mnt/glusterfs/somefile.txt")
>>> counts = file.flatMap(lambda line: line.split(" ")).map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b)
>>> counts.saveAsTextFile("/mnt/glusterfs/output")
```

