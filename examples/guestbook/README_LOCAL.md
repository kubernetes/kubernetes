## Running the cluster Locally using a Vagrant.

This example shows how to build a simple multi-tier web application using Kubernetes and Docker in a local environment. It is based on
the [guest book example](https://github.com/GoogleCloudPlatform/kubernetes/blob/master/examples/guestbook/README.md), so take a look at the one first to get familiarized with the basics about pods, controllers and services in Kubernetes.

The example combines a web frontend, a redis master for storage and a redis slave.

### Step Zero: Setup a local cluster using vagrant

The example will use a virtual machine using Vagrant to set up all the dependencies required by Kubernetes to run. Please make sure you
have [Vagrant](http://docs.vagrantup.com/v2/getting-started/index.html) installed and working before proceeding. Detailed instructions
about the Vagrant setup for Kubernetes could be found [here](https://github.com/GoogleCloudPlatform/kubernetes#getting-started-with-a-vagrant-cluster-on-your-host).

    $ cd kubernetes/
    $ ./hack/build-go.sh
    $ vagrant up

If the cluster started, you should be able to access it from the private IP of the VM in the port 8000. You could test it by executing the following from
your machine:

```
$ curl -X GET http://10.245.1.2:8080/api/v1beta1/pods
```

### Step Two: Turn up the redis master.

Go to the Kubernetes home directory and use the json example provided in examples/guestbook/redis-master.json
to create the pod:

```bash
$ curl -X POST http://10.245.1.2:8080/api/v1beta1/pods --data @examples/guestbook/redis-master.json
```

The previous command should return an output like the following:

```
{
"kind": "Status",
"status": "working",
"details": "1"
}
```

You can check the status of the operation in the following url: `http://10.245.1.2:8080/api/v1beta1/operations/1`

Once it stopped working and the desired state is shown, you should be able to proceed. Remember to be patient the first time you start a pod, Kubernetes will need to pull this image to the minon and it could take some time.

At any time you can check the status of the pod created while visiting the following url: `http://10.245.1.2:8080/api/v1beta1/pods`

### Step Two: Turn up the master service.

A Kubernetes 'service' is a named load balancer that proxies traffic to one or more containers. The services in a Kubernetes cluster are discoverable inside other containers via environment variables. Services find the containers to load balance based on pod labels.

The pod that you created in Step One has the label `name=redis-master`. The selector field of the service determines which pods will receive the traffic sent to the service.  Create a file named `redis-master-service.json` that contains:

```js
{
  "id": "redismaster",
  "port": 10000,
  "selector": {
    "name": "redis-master"
  }
}
```

You can create this service in your local cluster by executing:

```shell
$ curl -X POST http://10.245.1.2:8080/api/v1beta1/services --data @examples/guestbook/redis-master-service.json
```

This will cause all pods to see the redis master apparently running on localhost:10000 inside your cluster. You can check this by doing the following:

```shell
$ redis-cli -p 10000
$ KEYS *
```

You can see that even though the docker containter is running on the port 6379, it will be accesible via the port 10000 because of the service.

### Step Three: Turn up the frontend controller.

Set up redis slave:

```shell
$ curl -X POST http://10.245.1.2:8080/api/v1beta1/replicationControllers --data @examples/guestbook/redis-slave-controller.json
```

And also the redis slave service:

```shell
$ curl -X POST http://10.245.1.2:8080/api/v1beta1/services --data @examples/guestbook/redis-slave-service.json
```

### Step Four: Turn up the frontend controller.

Let's create a frontend pod that will use the redis master service to read and set keys in redis:

```shell
$ curl -X POST http://10.245.1.2:8080/api/v1beta1/replicationControllers --data @examples/guestbook/frontend-controller.json
```

At any step, it is important to wait for the operations to finish, otherwise you docker containers may not be ready yet.

Now, you just need to check where is running in the cluster the pod for the frontend controller:

```shell
$ KUBERNETES_PROVIDER=vagrant ./cluster/kubecfg.sh list /pods
```

The output should look something like this:

```
Name                                   Image(s)                   Host                Labels
----------                             ----------                 ----------          ----------
redis-master-2                         dockerfile/redis           10.245.2.2/         name=redis-master
892bd18e-16e1-11e4-ac06-0800279696e1   brendanburns/redis-slave   10.245.2.2/         name=redisslave,replicationController=redisSlaveController
dd4b37f1-16e3-11e4-ac06-0800279696e1   brendanburns/php-redis     10.245.2.2/         name=frontend,replicationController=frontendController
892b4e42-16e1-11e4-ac06-0800279696e1   brendanburns/redis-slave   10.245.2.3/         name=redisslave,replicationController=redisSlaveController
dd4b1025-16e3-11e4-ac06-0800279696e1   brendanburns/php-redis     10.245.2.3/         name=frontend,replicationController=frontendController
dd4b68ae-16e3-11e4-ac06-0800279696e1   brendanburns/php-redis     10.245.2.4/         name=frontend,replicationController=frontendController
```

You should be able to access the frontend controller by going to one of the URL listed above:

```
http://10.245.2.2:8000
```