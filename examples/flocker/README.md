<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


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

Read more about the [Flocker Cluster Architecture](https://docs.clusterhq.com/en/latest/concepts/architecture.html) at the [Flocker Documentation](https://docs.clusterhq.com/).

It is recommended to follow [Installing Flocker](https://docs.clusterhq.com/en/latest/install/index.html) and the instructions below to set-up the Flocker cluster to be used with Kubernetes.

#### Flocker Control Service

The Flocker Control Service should be installed manually on a host, . In the future, this may be deployed in pod(s) and exposed as a Kubernetes service.

#### Flocker Agent(s)

The Flocker Agents should be manually installed on *all* Kubernetes nodes. These agents are responsible for (de)attachment and (un)mounting and are therefore services that should be run with appropriate privileges on these hosts.

In order for the plugin to connect to Flocker (via REST API), several environment variables must be specified on *all* Kubernetes nodes. This may be specified in an init script for the node's Kubelet service, for example.

- `FLOCKER_CONTROL_SERVICE_HOST` should refer to the hostname of the Control Service
- `FLOCKER_CONTROL_SERVICE_PORT` should refer to the port of the Control Service (the API service defaults to 4523 but this must still be specified)

The following environment variables should refer to keys and certificates on the host that are specific to that host.

- `FLOCKER_CONTROL_SERVICE_CA_FILE` should refer to the full path to the cluster certificate file
- `FLOCKER_CONTROL_SERVICE_CLIENT_KEY_FILE` should refer to the full path to the key file for the API user
- `FLOCKER_CONTROL_SERVICE_CLIENT_CERT_FILE` should refer to the full path to the certificate file for the API user

More details regarding cluster authentication can be found at the documentation: [Flocker Cluster Security & Authentication](https://docs.clusterhq.com/en/latest/concepts/security.html) and [Configuring Cluster Authentication](https://docs.clusterhq.com/en/latest/config/configuring-authentication.html).

### Create a pod with a Flocker volume

**Note**: A new dataset must first be provisioned using the Flocker tools. For example, using the [Volumes CLI](https://docs.clusterhq.com/en/latest/labs/volumes-cli.html)), create a new dataset called 'my-flocker-vol' of size 10GB:

```sh
flocker-volumes create -m name=my-flocker-vol -s 10G
```

The following *volume* spec from the [example pod](flocker-pod.yml) illustrates how to use this Flocker dataset as a volume.

```yaml
  volumes:
    - name: www-root
      flocker:
        datasetName: my-flocker-vol
```

- **datasetName** is the unique name for the Flocker dataset and should match the *name* in the metadata.

Use `kubetctl` to create the pod.

```sh
$ kubectl create -f examples/flocker/flocker-pod.yml
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

Add an index.html inside this directory and use `curl` to see this HTML file served up by nginx.

```sh

$ curl ip

```





<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/flocker/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
