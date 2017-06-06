# Spark example

Following this example, you will create a functional [Apache
Spark](http://spark.apache.org/) cluster using Kubernetes and
[Docker](http://docker.io).

You will setup a Spark master service and a set of Spark workers using Spark's [standalone mode](http://spark.apache.org/docs/latest/spark-standalone.html).

For the impatient expert, jump straight to the [tl;dr](#tldr)
section.

### Sources

The Docker images are heavily based on https://github.com/mattf/docker-spark.
And are curated in https://github.com/kubernetes/application-images/tree/master/spark

The Spark UI Proxy is taken from https://github.com/aseigneurin/spark-ui-proxy.

The PySpark examples are taken from http://stackoverflow.com/questions/4114167/checking-if-a-number-is-a-prime-number-in-python/27946768#27946768

## Step Zero: Prerequisites

This example assumes

- You have a Kubernetes cluster installed and running.
- That you have installed the ```kubectl``` command line tool installed in your path and configured to talk to your Kubernetes cluster
- That your Kubernetes cluster is running [kube-dns](https://github.com/kubernetes/dns) or an equivalent integration.

Optionally, your Kubernetes cluster should be configured with a Loadbalancer integration (automatically configured via kube-up or GKE)

## Step One: Create namespace

```sh
$ kubectl create -f examples/spark/namespace-spark-cluster.yaml
```

Now list all namespaces:

```sh
$ kubectl get namespaces
NAME          LABELS             STATUS
default       <none>             Active
spark-cluster name=spark-cluster Active
```

To configure kubectl to work with our namespace, we will create a new context using our current context as a base:

```sh
$ CURRENT_CONTEXT=$(kubectl config view -o jsonpath='{.current-context}')
$ USER_NAME=$(kubectl config view -o jsonpath='{.contexts[?(@.name == "'"${CURRENT_CONTEXT}"'")].context.user}')
$ CLUSTER_NAME=$(kubectl config view -o jsonpath='{.contexts[?(@.name == "'"${CURRENT_CONTEXT}"'")].context.cluster}')
$ kubectl config set-context spark --namespace=spark-cluster --cluster=${CLUSTER_NAME} --user=${USER_NAME}
$ kubectl config use-context spark
```

## Step Two: Start your Master service

The Master [service](https://kubernetes.io/docs/user-guide/services.md) is the master service
for a Spark cluster.

Use the
[`examples/spark/spark-master-controller.yaml`](spark-master-controller.yaml)
file to create a
[replication controller](https://kubernetes.io/docs/user-guide/replication-controller.md)
running the Spark Master service.

```console
$ kubectl create -f examples/spark/spark-master-controller.yaml
replicationcontroller "spark-master-controller" created
```

Then, use the
[`examples/spark/spark-master-service.yaml`](spark-master-service.yaml) file to
create a logical service endpoint that Spark workers can use to access the
Master pod:

```console
$ kubectl create -f examples/spark/spark-master-service.yaml
service "spark-master" created
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

Once the master is started, we'll want to check the Spark WebUI. In order to access the Spark WebUI, we will deploy a [specialized proxy](https://github.com/aseigneurin/spark-ui-proxy). This proxy is neccessary to access worker logs from the Spark UI.

Deploy the proxy controller with [`examples/spark/spark-ui-proxy-controller.yaml`](spark-ui-proxy-controller.yaml):

```console
$ kubectl create -f examples/spark/spark-ui-proxy-controller.yaml
replicationcontroller "spark-ui-proxy-controller" created
```

We'll also need a corresponding Loadbalanced service for our Spark Proxy [`examples/spark/spark-ui-proxy-service.yaml`](spark-ui-proxy-service.yaml):

```console
$ kubectl create -f examples/spark/spark-ui-proxy-service.yaml
service "spark-ui-proxy" created
```

After creating the service, you should eventually get a loadbalanced endpoint:

```console
$ kubectl get svc spark-ui-proxy -o wide
 NAME             CLUSTER-IP    EXTERNAL-IP                                                              PORT(S)   AGE       SELECTOR
spark-ui-proxy   10.0.51.107   aad59283284d611e6839606c214502b5-833417581.us-east-1.elb.amazonaws.com   80/TCP    9m        component=spark-ui-proxy
```

The Spark UI in the above example output will be available at http://aad59283284d611e6839606c214502b5-833417581.us-east-1.elb.amazonaws.com

If your Kubernetes cluster is not equipped with a Loadbalancer integration, you will need to use the [kubectl proxy](https://kubernetes.io/docs/user-guide/accessing-the-cluster.md#using-kubectl-proxy) to
connect to the Spark WebUI:

```console
kubectl proxy --port=8001
```

At which point the UI will be available at
[http://localhost:8001/api/v1/proxy/namespaces/spark-cluster/services/spark-master:8080/](http://localhost:8001/api/v1/proxy/namespaces/spark-cluster/services/spark-master:8080/).

## Step Three: Start your Spark workers

The Spark workers do the heavy lifting in a Spark cluster. They
provide execution resources and data cache capabilities for your
program.

The Spark workers need the Master service to be running.

Use the [`examples/spark/spark-worker-controller.yaml`](spark-worker-controller.yaml) file to create a
[replication controller](https://kubernetes.io/docs/user-guide/replication-controller.md) that manages the worker pods.

```console
$ kubectl create -f examples/spark/spark-worker-controller.yaml
replicationcontroller "spark-worker-controller" created
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

## Step Four: Start the Zeppelin UI to launch jobs on your Spark cluster

The Zeppelin UI pod can be used to launch jobs into the Spark cluster either via
a web notebook frontend or the traditional Spark command line. See
[Zeppelin](https://zeppelin.incubator.apache.org/) and
[Spark architecture](https://spark.apache.org/docs/latest/cluster-overview.html)
for more details.

Deploy Zeppelin:

```console
$ kubectl create -f examples/spark/zeppelin-controller.yaml
replicationcontroller "zeppelin-controller" created
```

And the corresponding service:

```console
$ kubectl create -f examples/spark/zeppelin-service.yaml
service "zeppelin" created
```

Zeppelin needs the spark-master service to be running.

### Check to see if Zeppelin is running

```console
$ kubectl get pods -l component=zeppelin
NAME                        READY     STATUS    RESTARTS   AGE
zeppelin-controller-ja09s   1/1       Running   0          53s
```

## Step Five: Do something with the cluster

Now you have two choices, depending on your predilections. You can do something
graphical with the Spark cluster, or you can stay in the CLI.

For both choices, we will be working with this Python snippet:

```python
from math import sqrt; from itertools import count, islice

def isprime(n):
    return n > 1 and all(n%i for i in islice(count(2), int(sqrt(n)-1)))

nums = sc.parallelize(xrange(10000000))
print nums.filter(isprime).count()
```

### Do something fast with pyspark!

Simply copy and paste the python snippet into pyspark from within the zeppelin pod:

```console
$ kubectl exec zeppelin-controller-ja09s -it pyspark
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
>>> from math import sqrt; from itertools import count, islice
>>>
>>> def isprime(n):
...     return n > 1 and all(n%i for i in islice(count(2), int(sqrt(n)-1)))
...
>>> nums = sc.parallelize(xrange(10000000))

>>> print nums.filter(isprime).count()
664579
```

Congratulations, you now know how many prime numbers there are within the first 10 million numbers!

### Do something graphical and shiny!

Creating the Zeppelin service should have yielded you a Loadbalancer endpoint:

```console
$ kubectl get svc zeppelin -o wide
 NAME       CLUSTER-IP   EXTERNAL-IP                                                              PORT(S)   AGE       SELECTOR
zeppelin   10.0.154.1   a596f143884da11e6839506c114532b5-121893930.us-east-1.elb.amazonaws.com   80/TCP    3m        component=zeppelin
```

If your Kubernetes cluster does not have a Loadbalancer integration, then we will have to use port forwarding.

Take the Zeppelin pod from before and port-forward the WebUI port:

```console
$ kubectl port-forward zeppelin-controller-ja09s 8080:8080
```

This forwards `localhost` 8080 to container port 8080. You can then find
Zeppelin at [http://localhost:8080/](http://localhost:8080/).

Once you've loaded up the Zeppelin UI, create a "New Notebook". In there we will paste our python snippet, but we need to add a `%pyspark` hint for Zeppelin to understand it:

```
%pyspark
from math import sqrt; from itertools import count, islice

def isprime(n):
    return n > 1 and all(n%i for i in islice(count(2), int(sqrt(n)-1)))

nums = sc.parallelize(xrange(10000000))
print nums.filter(isprime).count()
```

After pasting in our code, press shift+enter or click the play icon to the right of our snippet. The Spark job will run and once again we'll have our result!

## Result

You now have services and replication controllers for the Spark master, Spark
workers and Spark driver.  You can take this example to the next step and start
using the Apache Spark cluster you just created, see
[Spark documentation](https://spark.apache.org/documentation.html) for more
information.

## tl;dr

```console
kubectl create -f examples/spark
```

After it's setup:

```console
kubectl get pods # Make sure everything is running
kubectl get svc -o wide # Get the Loadbalancer endpoints for spark-ui-proxy and zeppelin
```

At which point the Master UI and Zeppelin will be available at the URLs under the `EXTERNAL-IP` field.

You can also interact with the Spark cluster using the traditional `spark-shell` /
`spark-subsubmit` / `pyspark` commands by using `kubectl exec` against the
`zeppelin-controller` pod.

If your Kubernetes cluster does not have a Loadbalancer integration, use `kubectl proxy` and `kubectl port-forward` to access the Spark UI and Zeppelin.

For Spark UI:

```console
kubectl proxy --port=8001
```

Then visit [http://localhost:8001/api/v1/proxy/namespaces/spark-cluster/services/spark-ui-proxy/](http://localhost:8001/api/v1/proxy/namespaces/spark-cluster/services/spark-ui-proxy/).

For Zeppelin:

```console
kubectl port-forward zeppelin-controller-abc123 8080:8080 &
```

Then visit [http://localhost:8080/](http://localhost:8080/).

## Known Issues With Spark

* This provides a Spark configuration that is restricted to the cluster network,
  meaning the Spark master is only available as a cluster service. If you need
  to submit jobs using external client other than Zeppelin or `spark-submit` on
  the `zeppelin` pod, you will need to provide a way for your clients to get to
  the
  [`examples/spark/spark-master-service.yaml`](spark-master-service.yaml). See
  [Services](https://kubernetes.io/docs/user-guide/services.md) for more information.

## Known Issues With Zeppelin

* The Zeppelin pod is large, so it may take a while to pull depending on your
  network. The size of the Zeppelin pod is something we're working on, see issue #17231.

* Zeppelin may take some time (about a minute) on this pipeline the first time
  you run it. It seems to take considerable time to load.

* On GKE, `kubectl port-forward` may not be stable over long periods of time. If
  you see Zeppelin go into `Disconnected` state (there will be a red dot on the
  top right as well), the `port-forward` probably failed and needs to be
  restarted. See #12179.

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/spark/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
