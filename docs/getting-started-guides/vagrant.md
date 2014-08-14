## Getting started with Vagrant

### Prerequisites
1. Install latest version >= 1.6.2 of vagrant from http://www.vagrantup.com/downloads.html
2. Install latest version of Virtual Box from https://www.virtualbox.org/wiki/Downloads
3. Get the Kubernetes source:

```
git clone https://github.com/GoogleCloudPlatform/kubernetes.git
```

### Setup

By default, the Vagrant setup will create a single kubernetes-master and 3 kubernetes-minions.  You can control the number of minions that are instantiated via an environment variable on your host machine.  If you plan to work with replicas, we strongly encourage you to work with enough minions to satisfy your largest intended replica size.  If you do not plan to work with replicas, you can save some system resources by running with a single minion.

```
export KUBERNETES_NUM_MINIONS=3
```

To start your local cluster, open a terminal window and run:

```
cd kubernetes
vagrant up
```

Vagrant will provision each machine in the cluster with all the necessary components to build and run Kubernetes.  The initial setup can take a few minutes to complete on each machine.

By default, each VM in the cluster is running Fedora, and all of the Kubernetes services are installed into systemd.

To access the master or any minion:

```
vagrant ssh master
vagrant ssh minion-1
vagrant ssh minion-2
vagrant ssh minion-3
```

To view the service status and/or logs on the kubernetes-master:
```
vagrant ssh master
[vagrant@kubernetes-master ~] $ sudo systemctl status apiserver
[vagrant@kubernetes-master ~] $ sudo journalctl -r -u apiserver

[vagrant@kubernetes-master ~] $ sudo systemctl status controller-manager
[vagrant@kubernetes-master ~] $ sudo journalctl -r -u controller-manager

[vagrant@kubernetes-master ~] $ sudo systemctl status etcd
[vagrant@kubernetes-master ~] $ sudo systemctl status nginx
```

To view the services on any of the kubernetes-minion(s):
```
vagrant ssh minion-1
[vagrant@kubernetes-minion-1] $ sudo systemctl status docker
[vagrant@kubernetes-minion-1] $ sudo journalctl -r -u docker
[vagrant@kubernetes-minion-1] $ sudo systemctl status kubelet
[vagrant@kubernetes-minion-1] $ sudo journalctl -r -u kubelet
```

To push updates to new Kubernetes code after making source changes:
```
vagrant provision
```

To shutdown and then restart the cluster:
```
vagrant halt
vagrant up
```

To destroy the cluster:
```
vagrant destroy -f
```

You can also use the cluster/kube-*.sh scripts to interact with vagrant based providers just like any other hosting platform for kubernetes.

```
cd kubernetes
modify cluster/kube-env.sh:
  KUBERNETES_PROVIDER="vagrant"

## build the binary required by kubecfg.sh
hack/build-go.sh

cluster/kube-up.sh => brings up a vagrant cluster
cluster/kube-down.sh => destroys a vagrant cluster
cluster/kube-push.sh => updates a vagrant cluster
cluster/kubecfg.sh => interact with the cluster
```

### Authenticating with your master

To interact with the cluster, you must authenticate with the master when running cluster/kubecfg.sh commands.

If it's your first time using the cluster, your first invocation of cluster/kubecfg.sh will prompt you for credentials:

```
cd kubernetes
cluster/kubecfg.sh list minions
Please enter Username: vagrant
Please enter Password: vagrant
Minion identifier
----------
```

The kubecfg.sh command will cache your credentials in a .kubernetes_auth file so you will not be prompted in the future.
```
cat ~/.kubernetes_auth
{"User":"vagrant","Password":"vagrant"}
```

If you try Kubernetes against multiple cloud providers, make sure this file is correct for your target environment.

### Running a container

Your cluster is running, and you want to start running containers!

You can now use any of the cluster/kube-*.sh commands to interact with your VM machines.
```
cluster/kubecfg.sh list /pods
cluster/kubecfg.sh list /services
cluster/kubecfg.sh list /replicationControllers
cluster/kubecfg.sh -p 8080:80 run dockerfile/nginx 3 myNginx

## begin wait for provision to complete, you can monitor the minions by doing
  vagrant ssh minion-1
  sudo docker images
  ## you should see it pulling the dockerfile/nginx image, once the above command returns it
  sudo docker ps
  ## you should see your container running!
  exit
## end wait

## back on the host, introspect kubernetes!
cluster/kubecfg.sh list /pods
cluster/kubecfg.sh list /services
cluster/kubecfg.sh list /replicationControllers
```

Congratulations!

### Testing

The following will run all of the end-to-end testing scenarios assuming you set your environment in cluster/kube-env.sh

```
hack/e2e-test.sh
```


### Troubleshooting

#### I just created the cluster, but I am getting authorization errors!

You probably have an incorrect ~/.kubernetes_auth file for the cluster you are attempting to contact.

```
rm ~/.kubernetes_auth
```

And when using kubecfg.sh, provide the correct credentials:

```
Please enter Username: vagrant
Please enter Password: vagrant
```

#### I just created the cluster, but I do not see my container running!

If this is your first time creating the cluster, the kubelet on each minion schedules a number of docker pull requests to fetch prerequisite images.  This can take some time and as a result may delay your initial pod getting provisioned.

#### I changed Kubernetes code, but it's not running!

Are you sure there was no build error?  After running $ vagrant provision, scroll up and ensure that each Salt state was completed successfully on each box in the cluster.
It's very likely you see a build error due to an error in your source files!
