<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

Bare Metal Kubernetes on CoreOS with Calico Networking
------------------------------------------
This document describes how to deploy Kubernetes with Calico networking on _bare metal_ CoreOS. For more information on Project Calico, visit [projectcalico.org](http://projectcalico.org) and the [calico-containers repository](https://github.com/projectcalico/calico-containers).

To install Calico on an existing Kubernetes cluster, or for more information on deploying Calico with Kubernetes in a number of other environments take a look at our supported [deployment guides](https://github.com/projectcalico/calico-containers/tree/master/docs/cni/kubernetes).

Specifically, this guide will have you do the following:
- Deploy a Kubernetes master node on CoreOS using cloud-config.
- Deploy two Kubernetes compute nodes with Calico Networking using cloud-config.
- Configure `kubectl` to access your cluster.

The resulting cluster will use SSL between Kubernetes components.  It will run the SkyDNS service and kube-ui, and be fully conformant with the Kubernetes v1.1 conformance tests.

## Prerequisites and Assumptions

-   At least three bare-metal machines (or VMs) to work with. This guide will configure them as follows:
    - 1 Kubernetes Master
    - 2 Kubernetes Nodes
-   Your nodes should have IP connectivity to each other and the internet.
-   This guide assumes a DHCP server on your network to assign server IPs.
-   This guide uses `192.168.0.0/16` as the subnet from which pod IP addresses are assigned.  If this overlaps with your host subnet, you will need to configure Calico to use a different [IP pool](https://github.com/projectcalico/calico-containers/blob/master/docs/calicoctl/pool.md#calicoctl-pool-commands).

## Cloud-config

This guide will use [cloud-config](https://coreos.com/docs/cluster-management/setup/cloudinit-cloud-config/) to configure each of the nodes in our Kubernetes cluster.

We'll use two cloud-config files:
- `master-config.yaml`: cloud-config for the Kubernetes master
- `node-config.yaml`: cloud-config for each Kubernetes node

## Download CoreOS

Download the stable CoreOS bootable ISO from the [CoreOS website](https://coreos.com/docs/running-coreos/platforms/iso/).

## Configure the Kubernetes Master

1.  Once you've downloaded the ISO image, burn the ISO to a CD/DVD/USB key and boot from it (if using a virtual machine you can boot directly from the ISO).  Once booted, you should be automatically logged in as the `core` user at the terminal. At this point CoreOS is running from the ISO and it hasn't been installed yet.

2.  *On another machine*, download the the [master cloud-config template](https://raw.githubusercontent.com/projectcalico/calico-cni/k8s-1.1-docs/samples/kubernetes/cloud-config/master-config-template.yaml) and save it as `master-config.yaml`.

3.  Replace the following variables in the `master-config.yaml` file.

    - `<SSH_PUBLIC_KEY>`: The public key you will use for SSH access to this server. See [generating ssh keys](https://help.github.com/articles/generating-ssh-keys/)

4.  Copy the edited `master-config.yaml` to your Kubernetes master machine (using a USB stick, for example).

5.  The CoreOS bootable ISO comes with a tool called `coreos-install` which will allow us to install CoreOS and configure the machine using a cloud-config file.  The following command will download and install stable CoreOS using the `master-config.yaml` file we just created for configuration.  Run this on the Kubernetes master.

    > **Warning:** this is a destructive operation that erases disk `sda` on your server.

    ```
    sudo coreos-install -d /dev/sda -C stable -c master-config.yaml
    ```

6.  Once complete, restart the server and boot from `/dev/sda` (you may need to remove the ISO image). When it comes back up, you should have SSH access as the `core` user using the public key provided in the `master-config.yaml` file.

### Configure TLS

The master requires the CA certificate, `ca.pem`; its own certificate, `apiserver.pem` and its private key, `apiserver-key.pem`.  This [CoreOS guide](https://coreos.com/kubernetes/docs/latest/openssl.html) explains how to generate these.

1.  Generate the necessary certificates for the master. This [guide for generating Kubernetes TLS Assets](https://coreos.com/kubernetes/docs/latest/openssl.html) explains how to use OpenSSL to generate the required assets.

2.  Send the three files to your master host (using `scp` for example).

3.  Move them to the `/etc/kubernetes/ssl` folder and ensure that only the root user can read the key:

    ```
    # Move keys
    sudo mkdir -p /etc/kubernetes/ssl/
    sudo mv -t /etc/kubernetes/ssl/ ca.pem apiserver.pem apiserver-key.pem
    
    # Set Permissions
    sudo chmod 600 /etc/kubernetes/ssl/apiserver-key.pem
    sudo chown root:root /etc/kubernetes/ssl/apiserver-key.pem
    ```

4.  Restart the kubelet to pick up the changes:

    ```
    sudo systemctl restart kubelet
    ```

## Configure the compute nodes

The following steps will set up a single Kubernetes node for use as a compute host.  Run these steps to deploy each Kubernetes node in your cluster.

1.  Boot up the node machine using the bootable ISO we downloaded earlier.  You should be automatically logged in as the `core` user.

2.  Make a copy of the [node cloud-config template](https://raw.githubusercontent.com/projectcalico/calico-cni/k8s-1.1-docs/samples/kubernetes/cloud-config/node-config-template.yaml) for this machine.

3.  Replace the following placeholders in the `node-config.yaml` file to match your deployment.

    - `<HOSTNAME>`: Hostname for this node (e.g. kube-node1, kube-node2)
    - `<SSH_PUBLIC_KEY>`: The public key you will use for SSH access to this server.
    - `<KUBERNETES_MASTER>`: The IPv4 address of the Kubernetes master.

4.  Replace the following placeholders with the contents of their respective files.

    - `<CA_CERT>`: Complete contents of `ca.pem`
    - `<CA_KEY_CERT>`: Complete contents of `ca-key.pem`

    > **Important:** in a production deployment, embedding the secret key in cloud-config is a bad idea!  In production you should use an appropriate secret manager.

    > **Important:** Make sure you indent the entire file to match the indentation of the placeholder.  For example:
    >
    > ```
    >  - path: /etc/kubernetes/ssl/ca.pem
    >    owner: core
    >    permissions: 0644
    >    content: |
    >      <CA_CERT>
    > ```
    >
    > should look like this once the certificate is in place:
    >
    > ```
    >   - path: /etc/kubernetes/ssl/ca.pem
    >     owner: core
    >     permissions: 0644
    >     content: |
    >       -----BEGIN CERTIFICATE-----
    >       MIIC9zCCAd+gAwIBAgIJAJMnVnhVhy5pMA0GCSqGSIb3DQEBCwUAMBIxEDAOBgNV
    >       ...<snip>...
    >       QHwi1rNc8eBLNrd4BM/A1ZeDVh/Q9KxN+ZG/hHIXhmWKgN5wQx6/81FIFg==
    >       -----END CERTIFICATE-----
    > ```

5.  Move the modified `node-config.yaml` to your Kubernetes node machine and install and configure CoreOS on the node using the following command.

    > **Warning:** this is a destructive operation that erases disk `sda` on your server.

    ```
    sudo coreos-install -d /dev/sda -C stable -c node-config.yaml
    ```

6.  Once complete, restart the server and boot into `/dev/sda`. When it comes back up, you should have SSH access as the `core` user using the public key provided in the `node-config.yaml` file.  It will take some time for the node to be fully configured.

## Configure Kubeconfig

To administer your cluster from a separate host, you will need the client and admin certificates generated earlier (`ca.pem`, `admin.pem`, `admin-key.pem`). With certificates in place, run the following commands with the appropriate filepaths.

```
kubectl config set-cluster calico-cluster --server=https://<KUBERNETES_MASTER> --certificate-authority=<CA_CERT_PATH>
kubectl config set-credentials calico-admin --certificate-authority=<CA_CERT_PATH> --client-key=<ADMIN_KEY_PATH> --client-certificate=<ADMIN_CERT_PATH>
kubectl config set-context calico --cluster=calico-cluster --user=calico-admin
kubectl config use-context calico
```

Check your work with `kubectl get nodes`.

## Install the DNS Addon

Most Kubernetes deployments will require the DNS addon for service discovery. To install DNS, create the skydns service and replication controller provided.

```
kubectl create -f https://raw.githubusercontent.com/projectcalico/calico-cni/k8s-1.1-docs/samples/kubernetes/master/dns/skydns.yaml
```

## Install the Kubernetes UI Addon (Optional)

The Kubernetes UI can be installed using `kubectl` to run the following manifest file.

```
kubectl create -f https://raw.githubusercontent.com/projectcalico/calico-cni/k8s-1.1-docs/samples/kubernetes/master/kube-ui/kube-ui.yaml
```

## Launch other Services With Calico-Kubernetes

At this point, you have a fully functioning cluster running on Kubernetes with a master and two nodes networked with Calico. You can now follow any of the [standard documentation](../../../examples/) to set up other services on your cluster.

## Connectivity to outside the cluster

Because containers in this guide have private `192.168.0.0/16` IPs, you will need NAT to allow connectivity between containers and the internet. However, in a production data center deployment, NAT is not always necessary, since Calico can peer with the data center's border routers over BGP.

### NAT on the nodes

The simplest method for enabling connectivity from containers to the internet is to use outgoing NAT on your Kubernetes nodes.

Calico can provide outgoing NAT for containers.  To enable it, use the following `calicoctl` command:

```
ETCD_AUTHORITY=<master_ip:6666> calicoctl pool add <CONTAINER_SUBNET> --nat-outgoing
```

By default, `<CONTAINER_SUBNET>` will be `192.168.0.0/16`.  You can find out which pools have been configured with the following command:

```
ETCD_AUTHORITY=<master_ip:6666> calicoctl pool show
```

### NAT at the border router

In a data center environment, it is recommended to configure Calico to peer with the border routers over BGP. This means that the container IPs will be routable anywhere in the data center, and so NAT is not needed on the nodes (though it may be enabled at the data center edge to allow outbound-only internet connectivity).

The Calico documentation contains more information on how to configure Calico to [peer with existing infrastructure](https://github.com/projectcalico/calico-containers/blob/master/docs/ExternalConnectivity.md).

[![Analytics](https://ga-beacon.appspot.com/UA-52125893-3/kubernetes/docs/getting-started-guides/coreos/bare_metal_calico.md?pixel)](https://github.com/igrigorik/ga-beacon)



<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/coreos/bare_metal_calico.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
