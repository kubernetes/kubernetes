---
layout: docwithnav
title: "kubectl for docker users"
---
<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# kubectl for docker users

In this doc, we introduce the Kubernetes command line to for interacting with the api to docker-cli users. The tool, kubectl, is designed to be familiar to docker-cli users but there are a few necessary differences. Each section of this doc highlights a docker subcommand explains the kubectl equivalent.

**Table of Contents**
<!-- BEGIN MUNGE: GENERATED_TOC -->

- [kubectl for docker users](#kubectl-for-docker-users)
      - [docker run](#docker-run)
      - [docker ps](#docker-ps)
      - [docker exec](#docker-exec)
      - [docker logs](#docker-logs)
      - [docker stop and docker rm](#docker-stop-and-docker-rm)
      - [docker login](#docker-login)
      - [docker version](#docker-version)
      - [docker info](#docker-info)

<!-- END MUNGE: GENERATED_TOC -->

#### docker run

How do I run an nginx container and expose it to the world? Checkout [kubectl run](kubectl/kubectl_run.html).

With docker:

{% highlight console %}
{% raw %}
$ docker run -d --restart=always --name nginx-app -p 80:80 nginx
a9ec34d9878748d2f33dc20cb25c714ff21da8d40558b45bfaec9955859075d0
$ docker ps
CONTAINER ID        IMAGE               COMMAND                CREATED             STATUS              PORTS                         NAMES
a9ec34d98787        nginx               "nginx -g 'daemon of   2 seconds ago       Up 2 seconds        0.0.0.0:80->80/tcp, 443/tcp   nginx-app 
{% endraw %}
{% endhighlight %}

With kubectl:

{% highlight console %}
{% raw %}
# start the pod running nginx
$ kubectl run --image=nginx nginx-app
CONTROLLER   CONTAINER(S)   IMAGE(S)   SELECTOR        REPLICAS
nginx-app    nginx-app      nginx      run=nginx-app   1
# expose a port through with a service
$ kubectl expose rc nginx-app --port=80 --name=nginx-http
NAME         LABELS          SELECTOR        IP(S)     PORT(S)
nginx-http   run=nginx-app   run=nginx-app             80/TCP
{% endraw %}
{% endhighlight %}

With kubectl, we create a [replication controller](replication-controller.html) which will make sure that N pods are running nginx (where in is the number of replicas stated in the spec, which defaults to 1). We also create a [service](services.html) with a selector that matches the replication controller's selector. See the [quick-start.md](quick-start.html) guide for more information.

#### docker ps

How do I list what is currently running? Checkout [kubectl get](kubectl/kubectl_get.html).

With docker:

{% highlight console %}
{% raw %}
$ docker ps
CONTAINER ID        IMAGE               COMMAND                CREATED             STATUS              PORTS                         NAMES
a9ec34d98787        nginx               "nginx -g 'daemon of   About an hour ago   Up About an hour    0.0.0.0:80->80/tcp, 443/tcp   nginx-app
{% endraw %}
{% endhighlight %}

With kubectl:

{% highlight console %}
{% raw %}
$ kubectl get po
NAME              READY     STATUS    RESTARTS   AGE
nginx-app-5jyvm   1/1       Running   0          1h
{% endraw %}
{% endhighlight %}

#### docker exec

How do I execute a command in a container? Checkout [kubectl exec](kubectl/kubectl_exec.html).

With docker:

{% highlight console %}
{% raw %}
$ docker ps
CONTAINER ID        IMAGE               COMMAND                CREATED             STATUS              PORTS                         NAMES
a9ec34d98787        nginx               "nginx -g 'daemon of   8 minutes ago       Up 8 minutes        0.0.0.0:80->80/tcp, 443/tcp   nginx-app
$ docker exec a9ec34d98787 cat /etc/hostname
a9ec34d98787
{% endraw %}
{% endhighlight %}

With kubectl:

{% highlight console %}
{% raw %}
$ kubectl get po
NAME              READY     STATUS    RESTARTS   AGE
nginx-app-5jyvm   1/1       Running   0          10m
$ kubectl exec nginx-app-5jyvm -- cat /etc/hostname
nginx-app-5jyvm
{% endraw %}
{% endhighlight %}

What about interactive commands?


With docker:

{% highlight console %}
{% raw %}
$ docker exec -ti a9ec34d98787 /bin/sh
# exit
{% endraw %}
{% endhighlight %}

With kubectl:

{% highlight console %}
{% raw %}
$ kubectl exec -ti nginx-app-5jyvm -- /bin/sh      
# exit
{% endraw %}
{% endhighlight %}

For more information see [Getting into containers](getting-into-containers.html).

#### docker logs

How do I follow stdout/stderr of a running process? Checkout [kubectl logs](kubectl/kubectl_logs.html).


With docker:

{% highlight console %}
{% raw %}
$ docker logs -f a9e
192.168.9.1 - - [14/Jul/2015:01:04:02 +0000] "GET / HTTP/1.1" 200 612 "-" "curl/7.35.0" "-"
192.168.9.1 - - [14/Jul/2015:01:04:03 +0000] "GET / HTTP/1.1" 200 612 "-" "curl/7.35.0" "-"
{% endraw %}
{% endhighlight %}

With kubectl:

{% highlight console %}
{% raw %}
$ kubectl logs -f nginx-app-zibvs
10.240.63.110 - - [14/Jul/2015:01:09:01 +0000] "GET / HTTP/1.1" 200 612 "-" "curl/7.26.0" "-"
10.240.63.110 - - [14/Jul/2015:01:09:02 +0000] "GET / HTTP/1.1" 200 612 "-" "curl/7.26.0" "-"
{% endraw %}
{% endhighlight %}

Now's a good time to mention slight difference between pods and containers; by default pods will not terminate if their process's exit. Instead it will restart the process. This is similar to the docker run option `--restart=always` with one major difference. In docker, the output for each invocation of the process is concatenated but for Kubernetes, each invokation is separate. To see the output from a prevoius run in Kubernetes, do this:

{% highlight console %}
{% raw %}
$ kubectl logs --previous nginx-app-zibvs
10.240.63.110 - - [14/Jul/2015:01:09:01 +0000] "GET / HTTP/1.1" 200 612 "-" "curl/7.26.0" "-"
10.240.63.110 - - [14/Jul/2015:01:09:02 +0000] "GET / HTTP/1.1" 200 612 "-" "curl/7.26.0" "-"
{% endraw %}
{% endhighlight %}

See [Logging](logging.html) for more information.

#### docker stop and docker rm

How do I stop and delete a running process? Checkout [kubectl delete](kubectl/kubectl_delete.html).

With docker

{% highlight console %}
{% raw %}
$ docker ps
CONTAINER ID        IMAGE               COMMAND                CREATED             STATUS              PORTS                         NAMES
a9ec34d98787        nginx               "nginx -g 'daemon of   22 hours ago        Up 22 hours         0.0.0.0:80->80/tcp, 443/tcp   nginx-app
$ docker stop a9ec34d98787
a9ec34d98787
$ docker rm a9ec34d98787
a9ec34d98787
{% endraw %}
{% endhighlight %}

With kubectl:

{% highlight console %}
{% raw %}
$ kubectl get rc nginx-app
CONTROLLER   CONTAINER(S)   IMAGE(S)   SELECTOR        REPLICAS
nginx-app    nginx-app      nginx      run=nginx-app   1
$ kubectl get po
NAME              READY     STATUS    RESTARTS   AGE
nginx-app-aualv   1/1       Running   0          16s
$ kubectl delete rc nginx-app
NAME              READY     STATUS    RESTARTS   AGE
nginx-app-aualv   1/1       Running   0          16s
$ kubectl get po
NAME      READY     STATUS    RESTARTS   AGE
{% endraw %}
{% endhighlight %}

Notice that we don't delete the pod directly. With kubectl we want to delete the replication controller that owns the pod. If we delete the pod directly, the replication controller will recreate the pod.

#### docker login

There is no direct analog of 'docker login' in kubectl. If you are interested in using Kubernetes with a private registry, see [Using a Private Registry](images.html#using-a-private-registry).

#### docker version

How do I get the version of my client and server? Checkout [kubectl version](kubectl/kubectl_version.html).

With docker:

{% highlight console %}
{% raw %}
$ docker version
Client version: 1.7.0
Client API version: 1.19
Go version (client): go1.4.2
Git commit (client): 0baf609
OS/Arch (client): linux/amd64
Server version: 1.7.0
Server API version: 1.19
Go version (server): go1.4.2
Git commit (server): 0baf609
OS/Arch (server): linux/amd64
{% endraw %}
{% endhighlight %}

With kubectl:

{% highlight console %}
{% raw %}
$ kubectl version
Client Version: version.Info{Major:"0", Minor:"20.1", GitVersion:"v0.20.1", GitCommit:"", GitTreeState:"not a git tree"}
Server Version: version.Info{Major:"0", Minor:"21+", GitVersion:"v0.21.1-411-g32699e873ae1ca-dirty", GitCommit:"32699e873ae1caa01812e41de7eab28df4358ee4", GitTreeState:"dirty"}
{% endraw %}
{% endhighlight %}

#### docker info

How do I get miscellaneous info about my environment and configuration? Checkout [kubectl cluster-info](kubectl/kubectl_cluster-info.html).

With docker:

{% highlight console %}
{% raw %}
$ docker info
Containers: 40
Images: 168
Storage Driver: aufs
 Root Dir: /usr/local/google/docker/aufs
 Backing Filesystem: extfs
 Dirs: 248
 Dirperm1 Supported: false
Execution Driver: native-0.2
Logging Driver: json-file
Kernel Version: 3.13.0-53-generic
Operating System: Ubuntu 14.04.2 LTS
CPUs: 12
Total Memory: 31.32 GiB
Name: k8s-is-fun.mtv.corp.google.com
ID: ADUV:GCYR:B3VJ:HMPO:LNPQ:KD5S:YKFQ:76VN:IANZ:7TFV:ZBF4:BYJO
WARNING: No swap limit support
{% endraw %}
{% endhighlight %}

With kubectl:

{% highlight console %}
{% raw %}
$ kubectl cluster-info
Kubernetes master is running at https://108.59.85.141
KubeDNS is running at https://108.59.85.141/api/v1/proxy/namespaces/kube-system/services/kube-dns
KubeUI is running at https://108.59.85.141/api/v1/proxy/namespaces/kube-system/services/kube-ui
Grafana is running at https://108.59.85.141/api/v1/proxy/namespaces/kube-system/services/monitoring-grafana
Heapster is running at https://108.59.85.141/api/v1/proxy/namespaces/kube-system/services/monitoring-heapster
InfluxDB is running at https://108.59.85.141/api/v1/proxy/namespaces/kube-system/services/monitoring-influxdb
{% endraw %}
{% endhighlight %}


<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/docker-cli-to-kubectl.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

