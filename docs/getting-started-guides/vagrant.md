## Getting started with Vagrant

Running kubernetes with Vagrant (and VirtualBox) is an easy way to run/test/develop on your local machine (Linux, Mac OS X).

### Prerequisites
1. Install latest version >= 1.6.2 of vagrant from http://www.vagrantup.com/downloads.html
2. Install one of:
   1. The latest version of Virtual Box from https://www.virtualbox.org/wiki/Downloads
   2. [VMWare Fusion](https://www.vmware.com/products/fusion/) version 5 or greater as well as the appropriate [Vagrant VMWare Fusion provider](https://www.vagrantup.com/vmware)
   3. [VMWare Workstation](https://www.vmware.com/products/workstation/) version 9 or greater as well as the [Vagrant VMWare Workstation provider](https://www.vagrantup.com/vmware)
   4. [Parallels Desktop](https://www.parallels.com/products/desktop/) version 9 or greater as well as the [Vagrant Parallels provider](https://parallels.github.io/vagrant-parallels/)

### Setup

Setting up a cluster is as simple as running:

```sh
export KUBERNETES_PROVIDER=vagrant
curl -sS https://get.k8s.io | bash
```

The `KUBERNETES_PROVIDER` environment variable tells all of the various cluster management scripts which variant to use.  If you forget to set this, the assumption is you are running on Google Compute Engine.

By default, the Vagrant setup will create a single kubernetes-master and 1 kubernetes-minion. Each VM will take 1 GB, so make sure you have at least 2GB to 4GB of free memory (plus appropriate free disk space). To start your local cluster, open a shell and run:

```sh
cd kubernetes

export KUBERNETES_PROVIDER=vagrant
./cluster/kube-up.sh
```

Vagrant will provision each machine in the cluster with all the necessary components to run Kubernetes.  The initial setup can take a few minutes to complete on each machine.

If you installed more than one Vagrant provider, Kubernetes will usually pick the appropriate one. However, you can override which one Kubernetes will use by setting the [`VAGRANT_DEFAULT_PROVIDER`](https://docs.vagrantup.com/v2/providers/default.html) environment variable:

```sh
export VAGRANT_DEFAULT_PROVIDER=parallels
export KUBERNETES_PROVIDER=vagrant
./cluster/kube-up.sh
```

By default, each VM in the cluster is running Fedora, and all of the Kubernetes services are installed into systemd.

To access the master or any minion:

```sh
vagrant ssh master
vagrant ssh minion-1
```

If you are running more than one minion, you can access the others by:

```sh
vagrant ssh minion-2
vagrant ssh minion-3
```

To view the service status and/or logs on the kubernetes-master:
```sh
vagrant ssh master
[vagrant@kubernetes-master ~] $ sudo systemctl status kube-apiserver
[vagrant@kubernetes-master ~] $ sudo journalctl -r -u kube-apiserver

[vagrant@kubernetes-master ~] $ sudo systemctl status kube-controller-manager
[vagrant@kubernetes-master ~] $ sudo journalctl -r -u kube-controller-manager

[vagrant@kubernetes-master ~] $ sudo systemctl status etcd
[vagrant@kubernetes-master ~] $ sudo systemctl status nginx
```

To view the services on any of the kubernetes-minion(s):
```sh
vagrant ssh minion-1
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

You may need to build the binaries first, you can do this with ```make```

```sh
$ ./cluster/kubectl.sh get minions

NAME                LABELS
10.245.1.4          <none>
10.245.1.5          <none>
10.245.1.3          <none>
```

### Authenticating with your master

When using the vagrant provider in Kubernetes, the `cluster/kubectl.sh` script will cache your credentials in a `~/.kubernetes_vagrant_auth` file so you will not be prompted for them in the future.

```sh
cat ~/.kubernetes_vagrant_auth
{ "User": "vagrant",
  "Password": "vagrant",
  "CAFile": "/home/k8s_user/.kubernetes.vagrant.ca.crt",
  "CertFile": "/home/k8s_user/.kubecfg.vagrant.crt",
  "KeyFile": "/home/k8s_user/.kubecfg.vagrant.key"
}
```

You should now be set to use the `cluster/kubectl.sh` script. For example try to list the minions that you have started with:

```sh
./cluster/kubectl.sh get minions
```

### Running containers

Your cluster is running, you can list the minions in your cluster:

```sh
$ ./cluster/kubectl.sh get minions

NAME                 LABELS
10.245.2.4           <none>
10.245.2.3           <none>
10.245.2.2           <none>
```

Now start running some containers!

You can now use any of the `cluster/kube-*.sh` commands to interact with your VM machines.
Before starting a container there will be no pods, services and replication controllers.

```sh
$ ./cluster/kubectl.sh get pods
NAME   IMAGE(S)   HOST   LABELS   STATUS

$ ./cluster/kubectl.sh get services
NAME   LABELS   SELECTOR   IP   PORT

$ ./cluster/kubectl.sh get replicationControllers
NAME   IMAGE(S   SELECTOR   REPLICAS
```

Start a container running nginx with a replication controller and three replicas

```sh
$ ./cluster/kubectl.sh run-container my-nginx --image=nginx --replicas=3 --port=80
```

When listing the pods, you will see that three containers have been started and are in Waiting state:

```sh
$ ./cluster/kubectl.sh get pods
NAME                                   IMAGE(S)            HOST                    LABELS         STATUS
781191ff-3ffe-11e4-9036-0800279696e1   nginx               10.245.2.4/10.245.2.4   name=myNginx   Waiting
7813c8bd-3ffe-11e4-9036-0800279696e1   nginx               10.245.2.2/10.245.2.2   name=myNginx   Waiting
78140853-3ffe-11e4-9036-0800279696e1   nginx               10.245.2.3/10.245.2.3   name=myNginx   Waiting
```

You need to wait for the provisioning to complete, you can monitor the minions by doing:

```sh
$ sudo salt '*minion-1' cmd.run 'docker images'
kubernetes-minion-1:
    REPOSITORY          TAG                 IMAGE ID            CREATED             VIRTUAL SIZE
    <none>              <none>              96864a7d2df3        26 hours ago        204.4 MB
    google/cadvisor     latest              e0575e677c50        13 days ago         12.64 MB
    kubernetes/pause    latest              6c4579af347b        8 weeks ago         239.8 kB
```

Once the docker image for nginx has been downloaded, the container will start and you can list it:

```sh
$ sudo salt '*minion-1' cmd.run 'docker ps'
kubernetes-minion-1:
    CONTAINER ID        IMAGE                     COMMAND                CREATED             STATUS              PORTS                    NAMES
    dbe79bf6e25b        nginx:latest              "nginx"                21 seconds ago      Up 19 seconds                                k8s--mynginx.8c5b8a3a--7813c8bd_-_3ffe_-_11e4_-_9036_-_0800279696e1.etcd--7813c8bd_-_3ffe_-_11e4_-_9036_-_0800279696e1--fcfa837f
    fa0e29c94501        kubernetes/pause:latest   "/pause"               8 minutes ago       Up 8 minutes        0.0.0.0:8080->80/tcp     k8s--net.a90e7ce4--7813c8bd_-_3ffe_-_11e4_-_9036_-_0800279696e1.etcd--7813c8bd_-_3ffe_-_11e4_-_9036_-_0800279696e1--baf5b21b
    aa2ee3ed844a        google/cadvisor:latest    "/usr/bin/cadvisor -   38 minutes ago      Up 38 minutes                                k8s--cadvisor.9e90d182--cadvisor_-_agent.file--4626b3a2
    65a3a926f357        kubernetes/pause:latest   "/pause"               39 minutes ago      Up 39 minutes       0.0.0.0:4194->8080/tcp   k8s--net.c5ba7f0e--cadvisor_-_agent.file--342fd561
```

Going back to listing the pods, services and replicationControllers, you now have:

```sh
$ ./cluster/kubectl.sh get pods
NAME                                   IMAGE(S)            HOST                    LABELS         STATUS
781191ff-3ffe-11e4-9036-0800279696e1   nginx               10.245.2.4/10.245.2.4   name=myNginx   Running
7813c8bd-3ffe-11e4-9036-0800279696e1   nginx               10.245.2.2/10.245.2.2   name=myNginx   Running
78140853-3ffe-11e4-9036-0800279696e1   nginx               10.245.2.3/10.245.2.3   name=myNginx   Running

$ ./cluster/kubectl.sh get services
NAME   LABELS   SELECTOR   IP   PORT

$ ./cluster/kubectl.sh get replicationControllers
NAME      IMAGE(S            SELECTOR       REPLICAS
myNginx   nginx              name=my-nginx   3
```

We did not start any services, hence there are none listed. But we see three replicas displayed properly.
Check the [guestbook](../../examples/guestbook/README.md) application to learn how to create a service.
You can already play with resizing the replicas with:

```sh
$ ./cluster/kubectl.sh resize rc my-nginx --replicas=2
$ ./cluster/kubectl.sh get pods
NAME                                   IMAGE(S)            HOST                    LABELS         STATUS
7813c8bd-3ffe-11e4-9036-0800279696e1   nginx               10.245.2.2/10.245.2.2   name=myNginx   Running
78140853-3ffe-11e4-9036-0800279696e1   nginx               10.245.2.3/10.245.2.3   name=myNginx   Running
```

Congratulations!

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

```sh
cat ~/.kubernetes_vagrant_auth
{
  "User": "vagrant",
  "Password": "vagrant"
}
```

#### I just created the cluster, but I do not see my container running!

If this is your first time creating the cluster, the kubelet on each minion schedules a number of docker pull requests to fetch prerequisite images.  This can take some time and as a result may delay your initial pod getting provisioned.

#### I want to make changes to Kubernetes code!

To set up a vagrant cluster for hacking, follow the [vagrant developer guide](../devel/developer-guides/vagrant.md).

#### I have brought Vagrant up but the minions won't validate!

Log on to one of the minions (`vagrant ssh minion-1`) and inspect the salt minion log (`sudo cat /var/log/salt/minion`).

#### I want to change the number of minions!

You can control the number of minions that are instantiated via the environment variable `NUM_MINIONS` on your host machine.  If you plan to work with replicas, we strongly encourage you to work with enough minions to satisfy your largest intended replica size.  If you do not plan to work with replicas, you can save some system resources by running with a single minion. You do this, by setting `NUM_MINIONS` to 1 like so:

```sh
export NUM_MINIONS=1
```

#### I want my VMs to have more memory!

You can control the memory allotted to virtual machines with the `KUBERNETES_MEMORY` environment variable.
Just set it to the number of megabytes you would like the machines to have. For example:

```sh
export KUBERNETES_MEMORY=2048
```

If you need more granular control, you can set the amount of memory for the master and minions independently. For example:

```sh
export KUBERNETES_MASTER_MEMORY=1536
export KUBERNETES_MINION_MEMORY=2048
```

#### I ran vagrant suspend and nothing works!
```vagrant suspend``` seems to mess up the network.  It's not supported at this time.
