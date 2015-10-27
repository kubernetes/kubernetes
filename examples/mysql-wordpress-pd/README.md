<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Persistent Installation of MySQL and WordPress on Kubernetes

This example describes how to run a persistent installation of [Wordpress](https://wordpress.org/) using the [volumes](../../docs/user-guide/volumes.md) feature of Kubernetes, and [Google Compute Engine](https://cloud.google.com/compute/docs/disks) [persistent disks](../../docs/user-guide/volumes.md#gcepersistentdisk).

We'll use the [mysql](https://registry.hub.docker.com/_/mysql/) and [wordpress](https://registry.hub.docker.com/_/wordpress/) official [Docker](https://www.docker.com/) images for this installation. (The wordpress image includes an Apache server).

We'll create two Kubernetes [pods](../../docs/user-guide/pods.md) to run mysql and wordpress, both with associated persistent disks, then set up a Kubernetes [service](../../docs/user-guide/services.md) to front each pod.

This example demonstrates several useful things, including: how to set up and use persistent disks with Kubernetes pods; how to define Kubernetes services to leverage docker-links-compatible service environment variables; and use of an external load balancer to expose the wordpress service externally and make it transparent to the user if the wordpress pod moves to a different cluster node.

## Get started on Google Compute Engine (GCE)

Because we're using the `GCEPersistentDisk` type of volume for persistent storage, this example is only applicable to [Google Compute Engine](https://cloud.google.com/compute/). Take a look at the [volumes documentation](../../docs/user-guide/volumes.md) for other options.

First, if you have not already done so:

1. [Create](https://cloud.google.com/compute/docs/quickstart) a [Google Cloud Platform](https://cloud.google.com/) project.
2. [Enable billing](https://developers.google.com/console/help/new/#billing).
3. Install the [gcloud SDK](https://cloud.google.com/sdk/).

Authenticate with gcloud and set the gcloud default project name to point to the project you want to use for your Kubernetes cluster:

```sh
gcloud auth login
gcloud config set project <project-name>
```

Next, start up a Kubernetes cluster:

```sh
wget -q -O - https://get.k8s.io | bash
```

Please see the [GCE getting started guide](../../docs/getting-started-guides/gce.md) for full details and other options for starting a cluster.

## Create two persistent disks

For this WordPress installation, we're going to configure our Kubernetes [pods](../../docs/user-guide/pods.md) to use [persistent disks](https://cloud.google.com/compute/docs/disks). This means that we can preserve installation state across pod shutdown and re-startup.

You will need to create the disks in the same [GCE zone](https://cloud.google.com/compute/docs/zones) as the Kubernetes cluster. The default setup script will create the cluster in the `us-central1-b` zone, as seen in the [config-default.sh](../../cluster/gce/config-default.sh) file. Replace `$ZONE` below with the appropriate zone.

We will create two disks: one for the mysql pod, and one for the wordpress pod. In this example, we create 20GB disks, which will be sufficient for this demo. Feel free to change the size to align with your needs, as wordpress requirements can vary. Also, keep in mind that [disk performance scales with size](https://cloud.google.com/compute/docs/disks/#comparison_of_disk_types).

First create the mysql disk.

```sh
gcloud compute disks create --size=20GB --zone=$ZONE mysql-disk
```

Then create the wordpress disk.

```sh
gcloud compute disks create --size=20GB --zone=$ZONE wordpress-disk
```

## Start the Mysql Pod and Service

Now that the persistent disks are defined, the Kubernetes pods can be launched.  We'll start with the mysql pod.

### Start the Mysql pod

First, **edit [`mysql.yaml`](mysql.yaml)**, the mysql pod definition, to use a database password that you specify.
`mysql.yaml` looks like this:

<!-- BEGIN MUNGE: EXAMPLE mysql.yaml -->

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: mysql
  labels: 
    name: mysql
spec: 
  containers: 
    - resources:
        limits :
          cpu: 0.5
      image: mysql
      name: mysql
      env:
        - name: MYSQL_ROOT_PASSWORD
          # change this
          value: yourpassword
      ports: 
        - containerPort: 3306
          name: mysql
      volumeMounts:
          # name must match the volume name below
        - name: mysql-persistent-storage
          # mount path within the container
          mountPath: /var/lib/mysql
  volumes:
    - name: mysql-persistent-storage
      gcePersistentDisk:
        # This GCE PD must already exist.
        pdName: mysql-disk
        fsType: ext4
```

[Download example](mysql.yaml?raw=true)
<!-- END MUNGE: EXAMPLE mysql.yaml -->

Note that we've defined a volume mount for `/var/lib/mysql`, and specified a volume that uses the persistent disk (`mysql-disk`) that you created.
Once you've edited the file to set your database password, create the pod as follows, where `<kubernetes>` is the path to your Kubernetes installation:

```sh
$ kubectl create -f examples/mysql-wordpress-pd/mysql.yaml
```

It may take a short period before the new pod reaches the `Running` state.
List all pods to see the status of this new pod and the cluster node that it is running on:

```sh
$ kubectl get pods
```


#### Check the running pod on the Compute instance

You can take a look at the logs for a pod by using `kubectl.sh log`.  For example:

```sh
$ kubectl logs mysql
```

If you want to do deeper troubleshooting, e.g. if it seems a container is not staying up, you can also ssh in to the node that a pod is running on.  There, you can run `sudo -s`, then `docker ps -a` to see all the containers.  You can then inspect the logs of containers that have exited, via `docker logs <container_id>`.  (You can also find some relevant logs under `/var/log`, e.g. `docker.log` and `kubelet.log`).

### Start the Mysql service

We'll define and start a [service](../../docs/user-guide/services.md) that lets other pods access the mysql database on a known port and host.
We will specifically name the service `mysql`.  This will let us leverage the support for [Docker-links-compatible](../../docs/user-guide/services.md#how-do-they-work) service environment variables when we set up the wordpress pod. The wordpress Docker image expects to be linked to a mysql container named `mysql`, as you can see in the "How to use this image" section on the wordpress docker hub [page](https://registry.hub.docker.com/_/wordpress/).

So if we label our Kubernetes mysql service `mysql`, the wordpress pod will be able to use the Docker-links-compatible environment variables, defined by Kubernetes, to connect to the database.

The [`mysql-service.yaml`](mysql-service.yaml) file looks like this:

<!-- BEGIN MUNGE: EXAMPLE mysql-service.yaml -->

```yaml
apiVersion: v1
kind: Service
metadata: 
  labels: 
    name: mysql
  name: mysql
spec: 
  ports:
    # the port that this service should serve on
    - port: 3306
  # label keys and values that must match in order to receive traffic for this service
  selector: 
    name: mysql
```

[Download example](mysql-service.yaml?raw=true)
<!-- END MUNGE: EXAMPLE mysql-service.yaml -->

Start the service like this:

```sh
$ kubectl create -f examples/mysql-wordpress-pd/mysql-service.yaml
```

You can see what services are running via:

```sh
$ kubectl get services
```


## Start the WordPress Pod and Service

Once the mysql service is up, start the wordpress pod, specified in
[`wordpress.yaml`](wordpress.yaml).  Before you start it, **edit `wordpress.yaml`** and **set the database password to be the same as you used in `mysql.yaml`**.
Note that this config file also defines a volume, this one using the `wordpress-disk` persistent disk that you created.

<!-- BEGIN MUNGE: EXAMPLE wordpress.yaml -->

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: wordpress
  labels: 
    name: wordpress
spec: 
  containers: 
    - image: wordpress
      name: wordpress
      env:
        - name: WORDPRESS_DB_PASSWORD
          # change this - must match mysql.yaml password
          value: yourpassword
      ports: 
        - containerPort: 80
          name: wordpress
      volumeMounts:
          # name must match the volume name below
        - name: wordpress-persistent-storage
          # mount path within the container
          mountPath: /var/www/html
  volumes:
    - name: wordpress-persistent-storage
      gcePersistentDisk:
        # This GCE PD must already exist.
        pdName: wordpress-disk
        fsType: ext4
```

[Download example](wordpress.yaml?raw=true)
<!-- END MUNGE: EXAMPLE wordpress.yaml -->

Create the pod:

```sh
$ kubectl create -f examples/mysql-wordpress-pd/wordpress.yaml
```

And list the pods to check that the status of the new pod changes
to `Running`.  As above, this might take a minute.

```sh
$ kubectl get pods
```

### Start the WordPress service

Once the wordpress pod is running, start its service, specified by [`wordpress-service.yaml`](wordpress-service.yaml).

The service config file looks like this:

<!-- BEGIN MUNGE: EXAMPLE wordpress-service.yaml -->

```yaml
apiVersion: v1
kind: Service
metadata: 
  labels: 
    name: wpfrontend
  name: wpfrontend
spec: 
  ports:
    # the port that this service should serve on
    - port: 80
  # label keys and values that must match in order to receive traffic for this service
  selector: 
    name: wordpress
  type: LoadBalancer
```

[Download example](wordpress-service.yaml?raw=true)
<!-- END MUNGE: EXAMPLE wordpress-service.yaml -->

Note the `type: LoadBalancer` setting.  This will set up the wordpress service behind an external IP.
Note also that we've set the service port to 80.  We'll return to that shortly.

Start the service:

```sh
$ kubectl create -f examples/mysql-wordpress-pd/wordpress-service.yaml
```

and see it in the list of services:

```sh
$ kubectl get services
```

Then, find the external IP for your WordPress service by running:

```console
$ kubectl get services/wpfrontend
NAME                  CLUSTER_IP       EXTERNAL_IP       PORT(S)       SELECTOR               AGE
wpfrontend            10.0.0.2         1.2.3.4           80/TCP        ...                    ...
```

or by listing the forwarding rules for your project:

```sh
$ gcloud compute forwarding-rules list
```

Look for the rule called `wpfrontend`, which is what we named the wordpress service, and note its IP address.

## Visit your new WordPress blog

To access your new installation, you first may need to open up port 80 (the port specified in the wordpress service config) in the firewall for your cluster. You can do this, e.g. via:

```sh
$ gcloud compute firewall-rules create sample-http --allow tcp:80
```

This will define a firewall rule called `sample-http` that opens port 80 in the default network for your project.

Now, we can visit the running WordPress app.
Use the external IP that you obtained above, and visit it on port 80:

```
http://<external-ip>
```

You should see the familiar WordPress init page.

## Take down and restart your blog

Set up your WordPress blog and play around with it a bit.  Then, take down its pods and bring them back up again. Because you used persistent disks, your blog state will be preserved.

If you are just experimenting, you can take down and bring up only the pods:

```sh
$ kubectl delete -f examples/mysql-wordpress-pd/wordpress.yaml
$ kubectl delete -f examples/mysql-wordpress-pd/mysql.yaml
```

When you restart the pods again (using the `create` operation as described above), their services will pick up the new pods based on their labels.

If you want to shut down the entire app installation, you can delete the services as well.

If you are ready to turn down your Kubernetes cluster altogether, run:

```sh
$ cluster/kube-down.sh
```




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/mysql-wordpress-pd/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
