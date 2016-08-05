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

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

## Using Flocker volumes

[Flocker](https://clusterhq.com/flocker) is an open-source clustered container data volume manager. It provides management
and orchestration of data volumes backed by a variety of storage backends.

This example provides information about how to set-up a Flocker installation and configure it in Kubernetes, as well as how to use the plugin to use Flocker datasets as volumes in Kubernetes.

### Prerequisites

A Flocker cluster is required to use Flocker with Kubernetes. A Flocker cluster comprises:

- *Flocker Control Service*: provides a REST over HTTP API to modify the desired configuration of the cluster;
- *Flocker Dataset Agent(s)*: a convergence agent that modifies the cluster state to match the desired configuration;
- *Flocker Container Agent(s)*: a convergence agent that modifies the cluster state to match the desired configuration (unused in this configuration but still required in the cluster).

The Flocker cluster can be installed on the same nodes you are using for Kubernetes. For instance, you can install the Flocker Control Service on the same node as Kubernetes Master and Flocker Dataset/Container Agents on every Kubernetes Slave node.

It is recommended to follow [Installing Flocker](https://docs.clusterhq.com/en/latest/install/index.html) and the instructions below to set-up the Flocker cluster to be used with Kubernetes.

#### Flocker Control Service

The Flocker Control Service should be installed manually on a host. In the future, this may be deployed in pod(s) and exposed as a Kubernetes service.

#### Flocker Agent(s)

The Flocker Agents should be manually installed on *all* Kubernetes nodes. These agents are responsible for (de)attachment and (un)mounting and are therefore services that should be run with appropriate privileges on these hosts.

In order for the plugin to connect to Flocker (via REST API), several environment variables must be specified on *all* Kubernetes nodes. This may be specified in an init script for the node's Kubelet service, for example, you could store the below environment variables in a file called `/etc/flocker/env` and place `EnvironmentFile=/etc/flocker/env` into `/etc/systemd/system/kubelet.service` or wherever the `kubelet.service` file lives.

The environment variables that need to be set are:

- `FLOCKER_CONTROL_SERVICE_HOST` should refer to the hostname of the Control Service
- `FLOCKER_CONTROL_SERVICE_PORT` should refer to the port of the Control Service (the API service defaults to 4523 but this must still be specified)

The following environment variables should refer to keys and certificates on the host that are specific to that host.

- `FLOCKER_CONTROL_SERVICE_CA_FILE` should refer to the full path to the cluster certificate file
- `FLOCKER_CONTROL_SERVICE_CLIENT_KEY_FILE` should refer to the full path to the [api key](https://docs.clusterhq.com/en/latest/config/generate-api-plugin.html) file for the API user
- `FLOCKER_CONTROL_SERVICE_CLIENT_CERT_FILE` should refer to the full path to the [api certificate](https://docs.clusterhq.com/en/latest/config/generate-api-plugin.html) file for the API user

More details regarding cluster authentication can be found at the documentation: [Flocker Cluster Security & Authentication](https://docs.clusterhq.com/en/latest/concepts/security.html) and [Configuring Cluster Authentication](https://docs.clusterhq.com/en/latest/config/configuring-authentication.html).

### Create a pod with a Flocker volume

**Note**: A new dataset must first be provisioned using the Flocker tools or Docker CLI *(To use the Docker CLI, you need the [Flocker plugin for Docker](https://clusterhq.com/docker-plugin/) installed along with Docker 1.9+)*. For example, using the [Volumes CLI](https://docs.clusterhq.com/en/latest/labs/volumes-cli.html), create a new dataset called 'my-flocker-vol' of size 10GB:

```sh
flocker-volumes create -m name=my-flocker-vol -s 10G -n <node-uuid>

# -n or --node= Is the initial primary node for dataset (any unique 
# prefix of node uuid, see flocker-volumes list-nodes)
```

The following *volume* spec from the [example pod](flocker-pod.yml) illustrates how to use this Flocker dataset as a volume.

> Note, the [example pod](flocker-pod.yml) used here does not include a replication controller, therefore the POD will not be rescheduled upon failure. If your looking for an example that does include a replication controller and service spec you can use [this example pod including a replication controller](flocker-pod-with-rc.yml)

```yaml
  volumes:
    - name: www-root
      flocker:
        datasetName: my-flocker-vol
```

- **datasetName** is the unique name for the Flocker dataset and should match the *name* in the metadata.

Use `kubetctl` to create the pod.

```sh
$ kubectl create -f examples/volumes/flocker/flocker-pod.yml
```

You should now verify that the pod is running and determine it's IP address:

```sh
$ kubectl get pods
NAME             READY     STATUS    RESTARTS   AGE
flocker          1/1       Running   0          3m
$ kubectl get pods flocker -t '{{.status.hostIP}}{{"\n"}}'
172.31.25.62
```

An `ls` of the `/flocker` directory on the host (identified by the IP as above) will show the mount point for the volume.

```sh
$ ls /flocker
0cf8789f-00da-4da0-976a-b6b1dc831159
```

You can also see the mountpoint by inspecting the docker container on that host.

```sh
$ docker inspect -f "{{.Mounts}}" <container-id> | grep flocker
...{ /flocker/0cf8789f-00da-4da0-976a-b6b1dc831159 /usr/share/nginx/html true}
```

Add an index.html inside this directory and use `curl` to see this HTML file served up by nginx.

```sh
$ echo "<h1>Hello, World</h1>" | tee /flocker/0cf8789f-00da-4da0-976a-b6b1dc831159/index.html
$ curl ip

```

### More Info

Read more about the [Flocker Cluster Architecture](https://docs.clusterhq.com/en/latest/concepts/architecture.html) and learn more about Flocker by visiting the [Flocker Documentation](https://docs.clusterhq.com/).

#### Video Demo

To see a demo example of using Kubernetes and Flocker, visit [Flocker's blog post on High Availability with Kubernetes and Flocker](https://clusterhq.com/2015/12/22/ha-demo-kubernetes-flocker/)

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/volumes/flocker/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
