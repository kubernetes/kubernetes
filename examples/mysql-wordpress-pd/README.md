<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Persistent Installation of MySQL and WordPress on Kubernetes

This example describes how to run a persistent installation of
[WordPress](https://wordpress.org/) and
[MySQL](https://www.mysql.com/) on Kubernetes. We'll use the
[mysql](https://registry.hub.docker.com/_/mysql/) and
[wordpress](https://registry.hub.docker.com/_/wordpress/) official
[Docker](https://www.docker.com/) images for this installation. (The
WordPress image includes an Apache server).

Demonstrated Kubernetes Concepts:

* [Persistent Volumes](http://kubernetes.io/docs/user-guide/persistent-volumes/) to
  define persistent disks (disk lifecycle not tied to the Pods).
* [Services](http://kubernetes.io/docs/user-guide/services/) to enable Pods to
  locate one another.
* [External Load Balancers](http://kubernetes.io/docs/user-guide/services/#type-loadbalancer)
  to expose Services externally.
* [Deployments](http://kubernetes.io/docs/user-guide/deployments/) to ensure Pods
  stay up and running.
* [Secrets](http://kubernetes.io/docs/user-guide/secrets/) to store sensitive
  passwords.

## tl;dr Quickstart

Put your desired mysql password in a file called `password.txt` with
no trailing newline. The first `tr` command will remove the newline if
your editor added one.

```shell
tr --delete '\n' <password.txt >.strippedpassword.txt && mv .strippedpassword.txt password.txt
kubectl create -f https://raw.githubusercontent.com/kubernetes/kubernetes/master/examples/mysql-wordpress-pd/local-volumes.yaml
kubectl create secret generic mysql-pass --from-file=password.txt
kubectl create -f https://raw.githubusercontent.com/kubernetes/kubernetes/master/examples/mysql-wordpress-pd/mysql-deployment.yaml
kubectl create -f https://raw.githubusercontent.com/kubernetes/kubernetes/master/examples/mysql-wordpress-pd/wordpress-deployment.yaml
```

## Table of Contents

<!-- BEGIN MUNGE: GENERATED_TOC -->

- [Persistent Installation of MySQL and WordPress on Kubernetes](#persistent-installation-of-mysql-and-wordpress-on-kubernetes)
  - [tl;dr Quickstart](#tldr-quickstart)
  - [Table of Contents](#table-of-contents)
  - [Cluster Requirements](#cluster-requirements)
  - [Decide where you will store your data](#decide-where-you-will-store-your-data)
    - [Host Path](#host-path)
    - [GCE Persistent Disk](#gce-persistent-disk)
  - [Create the MySQL Password Secret](#create-the-mysql-password-secret)
  - [Deploy MySQL](#deploy-mysql)
  - [Deploy WordPress](#deploy-wordpress)
  - [Visit your new WordPress blog](#visit-your-new-wordpress-blog)
  - [Take down and restart your blog](#take-down-and-restart-your-blog)
  - [Next Steps](#next-steps)

<!-- END MUNGE: GENERATED_TOC -->

## Cluster Requirements

Kubernetes runs in a variety of environments and is inherently
modular. Not all clusters are the same. These are the requirements for
this example.

* Kubernetes version 1.2 is required due to using newer features, such
  at PV Claims and Deployments. Run `kubectl version` to see your
  cluster version.
* [Cluster DNS](../../build/kube-dns/) will be used for service discovery.
* An [external load balancer](http://kubernetes.io/docs/user-guide/services/#type-loadbalancer)
  will be used to access WordPress.
* [Persistent Volume Claims](http://kubernetes.io/docs/user-guide/persistent-volumes/)
  are used. You must create Persistent Volumes in your cluster to be
  claimed. This example demonstrates how to create two types of
  volumes, but any volume is sufficient.

Consult a
[Getting Started Guide](http://kubernetes.io/docs/getting-started-guides/)
to set up a cluster and the
[kubectl](http://kubernetes.io/docs/user-guide/prereqs/) command-line client.

## Decide where you will store your data

MySQL and WordPress will each use a
[Persistent Volume](http://kubernetes.io/docs/user-guide/persistent-volumes/)
to store their data. We will use a Persistent Volume Claim to claim an
available persistent volume. This example covers HostPath and
GCEPersistentDisk volumes. Choose one of the two, or see
[Types of Persistent Volumes](http://kubernetes.io/docs/user-guide/persistent-volumes/#types-of-persistent-volumes)
for more options.

### Host Path

Host paths are volumes mapped to directories on the host. **These
should be used for testing or single-node clusters only**. The data
will not be moved between nodes if the pod is recreated on a new
node. If the pod is deleted and recreated on a new node, data will be
lost.

Create the persistent volume objects in Kubernetes using
[local-volumes.yaml](local-volumes.yaml):

```shell
export KUBE_REPO=https://raw.githubusercontent.com/kubernetes/kubernetes/master
kubectl create -f $KUBE_REPO/examples/mysql-wordpress-pd/local-volumes.yaml
```

### GCE Persistent Disk

This storage option is applicable if you are running on
[Google Compute Engine](http://kubernetes.io/docs/getting-started-guides/gce/).

Create two persistent disks. You will need to create the disks in the
same [GCE zone](https://cloud.google.com/compute/docs/zones) as the
Kubernetes cluster. The default setup script will create the cluster
in the `us-central1-b` zone, as seen in the
[config-default.sh](../../cluster/gce/config-default.sh) file. Replace
`<zone>` below with the appropriate zone. The names `wordpress-1` and
`wordpress-2` must match the `pdName` fields we have specified in
[gce-volumes.yaml](gce-volumes.yaml).

```shell
gcloud compute disks create --size=20GB --zone=<zone> wordpress-1
gcloud compute disks create --size=20GB --zone=<zone> wordpress-2
```

Create the persistent volume objects in Kubernetes for those disks:

```shell
export KUBE_REPO=https://raw.githubusercontent.com/kubernetes/kubernetes/master
kubectl create -f $KUBE_REPO/examples/mysql-wordpress-pd/gce-volumes.yaml
```

## Create the MySQL Password Secret

Use a [Secret](http://kubernetes.io/docs/user-guide/secrets/) object
to store the MySQL password. First create a temporary file called
`password.txt` and save your password in it. Make sure to not have a
trailing newline at the end of the password. The first `tr` command
will remove the newline if your editor added one. Then, create the
Secret object.

```shell
tr --delete '\n' <password.txt >.strippedpassword.txt && mv .strippedpassword.txt password.txt
kubectl create secret generic mysql-pass --from-file=password.txt
```

This secret is referenced by the MySQL and WordPress pod configuration
so that those pods will have access to it. The MySQL pod will set the
database password, and the WordPress pod will use the password to
access the database.

## Deploy MySQL

Now that the persistent disks and secrets are defined, the Kubernetes
pods can be launched. Start MySQL using
[mysql-deployment.yaml](mysql-deployment.yaml).

```shell
kubectl create -f $KUBE_REPO/examples/mysql-wordpress-pd/mysql-deployment.yaml
```

Take a look at [mysql-deployment.yaml](mysql-deployment.yaml), and
note that we've defined a volume mount for `/var/lib/mysql`, and then
created a Persistent Volume Claim that looks for a 20G volume. This
claim is satisfied by any volume that meets the requirements, in our
case one of the volumes we created above.

Also look at the `env` section and see that we specified the password
by referencing the secret `mysql-pass` that we created above. Secrets
can have multiple key:value pairs. Ours has only one key
`password.txt` which was the name of the file we used to create the
secret. The [MySQL image](https://hub.docker.com/_/mysql/) sets the
database password using the `MYSQL_ROOT_PASSWORD` environment
variable.

It may take a short period before the new pod reaches the `Running`
state.  List all pods to see the status of this new pod.

```shell
kubectl get pods
```

```
NAME                          READY     STATUS    RESTARTS   AGE
wordpress-mysql-cqcf4-9q8lo   1/1       Running   0          1m
```

Kubernetes logs the stderr and stdout for each pod. Take a look at the
logs for a pod by using `kubectl log`. Copy the pod name from the
`get pods` command, and then:

```shell
kubectl logs <pod-name>
```

```
...
2016-02-19 16:58:05 1 [Note] InnoDB: 128 rollback segment(s) are active.
2016-02-19 16:58:05 1 [Note] InnoDB: Waiting for purge to start
2016-02-19 16:58:05 1 [Note] InnoDB: 5.6.29 started; log sequence number 1626007
2016-02-19 16:58:05 1 [Note] Server hostname (bind-address): '*'; port: 3306
2016-02-19 16:58:05 1 [Note] IPv6 is available.
2016-02-19 16:58:05 1 [Note]   - '::' resolves to '::';
2016-02-19 16:58:05 1 [Note] Server socket created on IP: '::'.
2016-02-19 16:58:05 1 [Warning] 'proxies_priv' entry '@ root@wordpress-mysql-cqcf4-9q8lo' ignored in --skip-name-resolve mode.
2016-02-19 16:58:05 1 [Note] Event Scheduler: Loaded 0 events
2016-02-19 16:58:05 1 [Note] mysqld: ready for connections.
Version: '5.6.29'  socket: '/var/run/mysqld/mysqld.sock'  port: 3306  MySQL Community Server (GPL)
```

Also in [mysql-deployment.yaml](mysql-deployment.yaml) we created a
service to allow other pods to reach this mysql instance. The name is
`wordpress-mysql` which resolves to the pod IP.

## Deploy WordPress

Next deploy WordPress using
[wordpress-deployment.yaml](wordpress-deployment.yaml):

```shell
kubectl create -f $KUBE_REPO/examples/mysql-wordpress-pd/wordpress-deployment.yaml
```

Here we are using many of the same features, such as a volume claim
for persistent storage and a secret for the password.

The [WordPress image](https://hub.docker.com/_/wordpress/) accepts the
database hostname through the environment variable
`WORDPRESS_DB_HOST`. We set the env value to the name of the MySQL
service we created: `wordpress-mysql`.

The WordPress service has the setting `type: LoadBalancer`.  This will
set up the wordpress service behind an external IP.

Find the external IP for your WordPress service. **It may take a minute
to have an external IP assigned to the service, depending on your
cluster environment.**

```shell
kubectl get services wordpress
```

```
NAME        CLUSTER-IP     EXTERNAL-IP     PORT(S)   AGE
wordpress   10.0.0.5       1.2.3.4         80/TCP    19h
```

## Visit your new WordPress blog

Now, we can visit the running WordPress app. Use the external IP of
the service that you obtained above.

```
http://<external-ip>
```

You should see the familiar WordPress init page.

![WordPress init page](WordPress.png "WordPress init page")

> Warning: Do not leave your WordPress installation on this page. If
> it is found by another user, they can set up a website on your
> instance and use it to serve potentially malicious content. You
> should either continue with the installation past the point at which
> you create your username and password, delete your instance, or set
> up a firewall to restrict access.

## Take down and restart your blog

Set up your WordPress blog and play around with it a bit. Then, take
down its pods and bring them back up again. Because you used
persistent disks, your blog state will be preserved.

All of the resources are labeled with `app=wordpress`, so you can
easily bring them down using a label selector:

```shell
kubectl delete deployment,service -l app=wordpress
kubectl delete secret mysql-pass
```

Later, re-creating the resources with the original commands will pick
up the original disks with all your data intact. Because we did not
delete the PV Claims, no other pods in the cluster could claim them
after we deleted our pods. Keeping the PV Claims also ensured
recreating the Pods did not cause the PD to switch Pods.

If you are ready to release your persistent volumes and the data on them, run:

```shell
kubectl delete pvc -l app=wordpress
```

And then delete the volume objects themselves:

```shell
kubectl delete pv local-pv-1 local-pv-2
```

or

```shell
kubectl delete pv wordpress-pv-1 wordpress-pv-2
```

## Next Steps

* [Introspection and Debugging](http://kubernetes.io/docs/user-guide/introspection-and-debugging/)
* [Jobs](http://kubernetes.io/docs/user-guide/jobs/) may be useful to run SQL queries.
* [Exec](http://kubernetes.io/docs/user-guide/getting-into-containers/)
* [Port Forwarding](http://kubernetes.io/docs/user-guide/connecting-to-applications-port-forward/)



<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/mysql-wordpress-pd/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
