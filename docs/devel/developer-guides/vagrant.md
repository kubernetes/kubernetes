<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

## Getting started with Vagrant

Running kubernetes with Vagrant (and VirtualBox) is an easy way to run/test/develop on your local machine (Linux, Mac OS X).

### Prerequisites

1. Install latest version >= 1.6.2 of vagrant from http://www.vagrantup.com/downloads.html
2. Install one of:
   1. The latest version of Virtual Box from https://www.virtualbox.org/wiki/Downloads
   2. [VMWare Fusion](https://www.vmware.com/products/fusion/) version 5 or greater as well as the appropriate [Vagrant VMWare Fusion provider](https://www.vagrantup.com/vmware)
   3. [VMWare Workstation](https://www.vmware.com/products/workstation/) version 9 or greater as well as the [Vagrant VMWare Workstation provider](https://www.vagrantup.com/vmware)
   4. [Parallels Desktop](https://www.parallels.com/products/desktop/) version 9 or greater as well as the [Vagrant Parallels provider](https://parallels.github.io/vagrant-parallels/)
3. Get or build a [binary release](../../../docs/getting-started-guides/binary_release.md)

### Setup

By default, the Vagrant setup will create a single master VM (called kubernetes-master) and one node (called kubernetes-minion-1). Each VM will take 1 GB, so make sure you have at least 2GB to 4GB of free memory (plus appropriate free disk space). To start your local cluster, open a shell and run:

```sh
cd kubernetes

export KUBERNETES_PROVIDER=vagrant
./cluster/kube-up.sh
```

The `KUBERNETES_PROVIDER` environment variable tells all of the various cluster management scripts which variant to use.  If you forget to set this, the assumption is you are running on Google Compute Engine.

If you installed more than one Vagrant provider, Kubernetes will usually pick the appropriate one. However, you can override which one Kubernetes will use by setting the [`VAGRANT_DEFAULT_PROVIDER`](https://docs.vagrantup.com/v2/providers/default.html) environment variable:

```sh
export VAGRANT_DEFAULT_PROVIDER=parallels
export KUBERNETES_PROVIDER=vagrant
./cluster/kube-up.sh
```

Vagrant will provision each machine in the cluster with all the necessary components to run Kubernetes.  The initial setup can take a few minutes to complete on each machine.

By default, each VM in the cluster is running Fedora, and all of the Kubernetes services are installed into systemd.

To access the master or any node:

```sh
vagrant ssh master
vagrant ssh minion-1
```

If you are running more than one nodes, you can access the others by:

```sh
vagrant ssh minion-2
vagrant ssh minion-3
```

To view the service status and/or logs on the kubernetes-master:

```console
$ vagrant ssh master
[vagrant@kubernetes-master ~] $ sudo systemctl status kube-apiserver
[vagrant@kubernetes-master ~] $ sudo journalctl -r -u kube-apiserver

[vagrant@kubernetes-master ~] $ sudo systemctl status kube-controller-manager
[vagrant@kubernetes-master ~] $ sudo journalctl -r -u kube-controller-manager

[vagrant@kubernetes-master ~] $ sudo systemctl status etcd
[vagrant@kubernetes-master ~] $ sudo systemctl status nginx
```

To view the services on any of the nodes:

```console
$ vagrant ssh minion-1
[vagrant@kubernetes-minion-1] $ sudo systemctl status docker
[vagrant@kubernetes-minion-1] $ sudo journalctl -r -u docker
[vagrant@kubernetes-minion-1] $ sudo systemctl status kubelet
[vagrant@kubernetes-minion-1] $ sudo journalctl -r -u kubelet
```

### Interacting with your Kubernetes cluster with Vagrant.

With your Kubernetes cluster up, you can manage the nodes in your cluster with the regular Vagrant commands.

To push updates to new Kubernetes code after making source changes:

```sh
./cluster/kube-push.sh
```

To stop and then restart the cluster:

```sh
vagrant halt
./cluster/kube-up.sh
```

To destroy the cluster:

```sh
vagrant destroy
```

Once your Vagrant machines are up and provisioned, the first thing to do is to check that you can use the `kubectl.sh` script.

You may need to build the binaries first, you can do this with `make`

```console
$ ./cluster/kubectl.sh get nodes

NAME                     LABELS                                          STATUS
kubernetes-minion-0whl   kubernetes.io/hostname=kubernetes-minion-0whl   Ready
kubernetes-minion-4jdf   kubernetes.io/hostname=kubernetes-minion-4jdf   Ready
kubernetes-minion-epbe   kubernetes.io/hostname=kubernetes-minion-epbe   Ready
```

### Interacting with your Kubernetes cluster with the `kube-*` scripts.

Alternatively to using the vagrant commands, you can also use the `cluster/kube-*.sh` scripts to interact with the vagrant based provider just like any other hosting platform for kubernetes.

All of these commands assume you have set `KUBERNETES_PROVIDER` appropriately:

```sh
export KUBERNETES_PROVIDER=vagrant
```

Bring up a vagrant cluster

```sh
./cluster/kube-up.sh
```

Destroy the vagrant cluster

```sh
./cluster/kube-down.sh
```

Update the vagrant cluster after you make changes (only works when building your own releases locally):

```sh
./cluster/kube-push.sh
```

Interact with the cluster

```sh
./cluster/kubectl.sh
```

### Authenticating with your master

When using the vagrant provider in Kubernetes, the `cluster/kubectl.sh` script will cache your credentials in a `~/.kubernetes_vagrant_auth` file so you will not be prompted for them in the future.

```console
$ cat ~/.kubernetes_vagrant_auth
{ "User": "vagrant",
  "Password": "vagrant"
  "CAFile": "/home/k8s_user/.kubernetes.vagrant.ca.crt",
  "CertFile": "/home/k8s_user/.kubecfg.vagrant.crt",
  "KeyFile": "/home/k8s_user/.kubecfg.vagrant.key"
}
```

You should now be set to use the `cluster/kubectl.sh` script. For example try to list the nodes that you have started with:

```sh
./cluster/kubectl.sh get nodes
```

### Running containers

Your cluster is running, you can list the nodes in your cluster:

```console
$ ./cluster/kubectl.sh get nodes

NAME                     LABELS                                          STATUS
kubernetes-minion-0whl   kubernetes.io/hostname=kubernetes-minion-0whl   Ready
kubernetes-minion-4jdf   kubernetes.io/hostname=kubernetes-minion-4jdf   Ready
kubernetes-minion-epbe   kubernetes.io/hostname=kubernetes-minion-epbe   Ready
```

Now start running some containers!

You can now use any of the cluster/kube-*.sh commands to interact with your VM machines.
Before starting a container there will be no pods, services and replication controllers.

```console
$ cluster/kubectl.sh get pods
NAME  READY   STATUS    RESTARTS    AGE

$ cluster/kubectl.sh get services
NAME  LABELS   SELECTOR    IP(S)    PORT(S)

$ cluster/kubectl.sh get rc
CONTROLLER  CONTAINER(S)   IMAGE(S)    SELECTOR    REPLICAS
```

Start a container running nginx with a replication controller and three replicas

```console
$ cluster/kubectl.sh run my-nginx --image=nginx --replicas=3 --port=80
CONTROLLER   CONTAINER(S)   IMAGE(S)   SELECTOR       REPLICAS
my-nginx     my-nginx       nginx      run=my-nginx   3
```

When listing the pods, you will see that three containers have been started and are in Waiting state:

```console
$ cluster/kubectl.sh get pods
NAME              READY     STATUS    RESTARTS   AGE
my-nginx-389da    1/1       Waiting   0          33s
my-nginx-kqdjk    1/1       Waiting   0          33s
my-nginx-nyj3x    1/1       Waiting   0          33s
```

You need to wait for the provisioning to complete, you can monitor the minions by doing:

```console
$ sudo salt '*minion-1' cmd.run 'docker images'
kubernetes-minion-1:
    REPOSITORY          TAG                 IMAGE ID            CREATED             VIRTUAL SIZE
    <none>              <none>              96864a7d2df3        26 hours ago        204.4 MB
    kubernetes/pause    latest              6c4579af347b        8 weeks ago         239.8 kB
```

Once the docker image for nginx has been downloaded, the container will start and you can list it:

```console
$ sudo salt '*minion-1' cmd.run 'docker ps'
kubernetes-minion-1:
    CONTAINER ID        IMAGE                     COMMAND                CREATED             STATUS              PORTS                    NAMES
    dbe79bf6e25b        nginx:latest              "nginx"                21 seconds ago      Up 19 seconds                                k8s--mynginx.8c5b8a3a--7813c8bd_-_3ffe_-_11e4_-_9036_-_0800279696e1.etcd--7813c8bd_-_3ffe_-_11e4_-_9036_-_0800279696e1--fcfa837f
    fa0e29c94501        kubernetes/pause:latest   "/pause"               8 minutes ago       Up 8 minutes        0.0.0.0:8080->80/tcp     k8s--net.a90e7ce4--7813c8bd_-_3ffe_-_11e4_-_9036_-_0800279696e1.etcd--7813c8bd_-_3ffe_-_11e4_-_9036_-_0800279696e1--baf5b21b
```

Going back to listing the pods, services and replicationcontrollers, you now have:

```console
$ cluster/kubectl.sh get pods
NAME              READY     STATUS    RESTARTS   AGE
my-nginx-389da    1/1       Running   0          33s
my-nginx-kqdjk    1/1       Running   0          33s
my-nginx-nyj3x    1/1       Running   0          33s

$ cluster/kubectl.sh get services
NAME   LABELS   SELECTOR   IP(S)   PORT(S)

$ cluster/kubectl.sh get rc
NAME        IMAGE(S)          SELECTOR       REPLICAS
my-nginx    nginx             run=my-nginx   3
```

We did not start any services, hence there are none listed. But we see three replicas displayed properly.
Check the [guestbook](../../../examples/guestbook/README.md) application to learn how to create a service.
You can already play with scaling the replicas with:

```console
$ ./cluster/kubectl.sh scale rc my-nginx --replicas=2
$ ./cluster/kubectl.sh get pods
NAME              READY     STATUS    RESTARTS   AGE
my-nginx-kqdjk    1/1       Running   0          13m
my-nginx-nyj3x    1/1       Running   0          13m
```

Congratulations!

### Testing

The following will run all of the end-to-end testing scenarios assuming you set your environment in `cluster/kube-env.sh`:

```sh
NUM_MINIONS=3 hack/e2e-test.sh
```

### Troubleshooting

#### I keep downloading the same (large) box all the time!

By default the Vagrantfile will download the box from S3. You can change this (and cache the box locally) by providing a name and an alternate URL when calling `kube-up.sh`

```sh
export KUBERNETES_BOX_NAME=choose_your_own_name_for_your_kuber_box
export KUBERNETES_BOX_URL=path_of_your_kuber_box
export KUBERNETES_PROVIDER=vagrant
./cluster/kube-up.sh
```

#### I just created the cluster, but I am getting authorization errors!

You probably have an incorrect ~/.kubernetes_vagrant_auth file for the cluster you are attempting to contact.

```sh
rm ~/.kubernetes_vagrant_auth
```

After using kubectl.sh make sure that the correct credentials are set:

```console
$ cat ~/.kubernetes_vagrant_auth
{
  "User": "vagrant",
  "Password": "vagrant"
}
```

#### I just created the cluster, but I do not see my container running!

If this is your first time creating the cluster, the kubelet on each node schedules a number of docker pull requests to fetch prerequisite images.  This can take some time and as a result may delay your initial pod getting provisioned.

#### I changed Kubernetes code, but it's not running!

Are you sure there was no build error?  After running `$ vagrant provision`, scroll up and ensure that each Salt state was completed successfully on each box in the cluster.
It's very likely you see a build error due to an error in your source files!

#### I have brought Vagrant up but the nodes won't validate!

Are you sure you built a release first? Did you install `net-tools`? For more clues, login to one of the nodes (`vagrant ssh minion-1`) and inspect the salt minion log (`sudo cat /var/log/salt/minion`).

#### I want to change the number of nodes!

You can control the number of nodes that are instantiated via the environment variable `NUM_MINIONS` on your host machine.  If you plan to work with replicas, we strongly encourage you to work with enough nodes to satisfy your largest intended replica size.  If you do not plan to work with replicas, you can save some system resources by running with a single node. You do this, by setting `NUM_MINIONS` to 1 like so:

```sh
export NUM_MINIONS=1
```

#### I want my VMs to have more memory!

You can control the memory allotted to virtual machines with the `KUBERNETES_MEMORY` environment variable.
Just set it to the number of megabytes you would like the machines to have. For example:

```sh
export KUBERNETES_MEMORY=2048
```

If you need more granular control, you can set the amount of memory for the master and nodes independently. For example:

```sh
export KUBERNETES_MASTER_MEMORY=1536
export KUBERNETES_MINION_MEMORY=2048
```

#### I ran vagrant suspend and nothing works!

`vagrant suspend` seems to mess up the network.  It's not supported at this time.




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/developer-guides/vagrant.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
