<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.1/docs/getting-started-guides/ubuntu-calico.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
Bare Metal Kubernetes with Calico Networking
------------------------------------------------

## Introduction

This document describes how to deploy Kubernetes with Calico networking from scratch on _bare metal_ Ubuntu. For more information on Project Calico, visit [projectcalico.org](http://projectcalico.org) and the [calico-containers repository](https://github.com/projectcalico/calico-containers).

To install Calico on an existing Kubernetes cluster, or for more information on deploying Calico with Kubernetes in a number of other environments take a look at our supported [deployment guides](https://github.com/projectcalico/calico-containers/tree/master/docs/cni/kubernetes).

This guide will set up a simple Kubernetes cluster with a single Kubernetes master and two Kubernetes nodes.  We'll run Calico's etcd cluster on the master and install the Calico daemon on the master and nodes.

## Prerequisites and Assumptions

- This guide uses `systemd` for process management. Ubuntu 15.04 supports systemd natively as do a number of other Linux distributions.
- All machines should have Docker >= 1.7.0 installed.
	- To install Docker on Ubuntu, follow [these instructions](https://docs.docker.com/installation/ubuntulinux/)
- All machines should have connectivity to each other and the internet.
- This guide assumes a DHCP server on your network to assign server IPs.
- This guide uses `192.168.0.0/16` as the subnet from which pod IP addresses are assigned.  If this overlaps with your host subnet, you will need to configure Calico to use a different [IP pool](https://github.com/projectcalico/calico-containers/blob/master/docs/calicoctl/pool.md#calicoctl-pool-commands).
- This guide assumes that none of the hosts have been configured with any Kubernetes or Calico software.
- This guide will set up a secure, TLS-authenticated API server.

## Set up the master

### Configure TLS

The master requires the root CA public key, `ca.pem`; the apiserver certificate, `apiserver.pem` and its private key, `apiserver-key.pem`.

1.  Create the file `openssl.cnf` with the following contents.

    ```
    [req]
    req_extensions = v3_req
    distinguished_name = req_distinguished_name
    [req_distinguished_name]
    [ v3_req ]
    basicConstraints = CA:FALSE
    keyUsage = nonRepudiation, digitalSignature, keyEncipherment
    subjectAltName = @alt_names
    [alt_names]
    DNS.1 = kubernetes
    DNS.2 = kubernetes.default
    IP.1 = 10.100.0.1 
    IP.2 = ${MASTER_IPV4}
    ```

> Replace ${MASTER_IPV4} with the Master's IP address on which the Kubernetes API will be accessible.

2.  Generate the necessary TLS assets.

    ```
    # Generate the root CA.
    openssl genrsa -out ca-key.pem 2048
    openssl req -x509 -new -nodes -key ca-key.pem -days 10000 -out ca.pem -subj "/CN=kube-ca"

    # Generate the API server keypair.
    openssl genrsa -out apiserver-key.pem 2048
    openssl req -new -key apiserver-key.pem -out apiserver.csr -subj "/CN=kube-apiserver" -config openssl.cnf
    openssl x509 -req -in apiserver.csr -CA ca.pem -CAkey ca-key.pem -CAcreateserial -out apiserver.pem -days 365 -extensions v3_req -extfile openssl.cnf
    ```

3.  You should now have the following three files: `ca.pem`, `apiserver.pem`, and `apiserver-key.pem`.  Send the three files to your master host (using `scp` for example).
4.  Move them to the `/etc/kubernetes/ssl` folder and ensure that only the root user can read the key:

    ```
    # Move keys
    sudo mkdir -p /etc/kubernetes/ssl/
    sudo mv -t /etc/kubernetes/ssl/ ca.pem apiserver.pem apiserver-key.pem
    
    # Set permissions
    sudo chmod 600 /etc/kubernetes/ssl/apiserver-key.pem
    sudo chown root:root /etc/kubernetes/ssl/apiserver-key.pem
    ```

### Install Kubernetes on the Master

We'll use the `kubelet` to bootstrap the Kubernetes master.

1.  Download and install the `kubelet` and `kubectl` binaries:

    ```
    sudo wget -N -P /usr/bin http://storage.googleapis.com/kubernetes-release/release/v1.1.4/bin/linux/amd64/kubectl
    sudo wget -N -P /usr/bin http://storage.googleapis.com/kubernetes-release/release/v1.1.4/bin/linux/amd64/kubelet
    sudo chmod +x /usr/bin/kubelet /usr/bin/kubectl
    ```

2.  Install the `kubelet` systemd unit file and start the `kubelet`:

    ```
    # Install the unit file
    sudo wget -N -P /etc/systemd https://raw.githubusercontent.com/projectcalico/calico-cni/k8s-1.1-docs/samples/kubernetes/master/kubelet.service

    # Enable the unit file so that it runs on boot
    sudo systemctl enable /etc/systemd/kubelet.service

    # Start the kubelet service
    sudo systemctl start kubelet.service
    ```

3.  Download and install the master manifest file, which will start the Kubernetes master services automatically:

    ```
    sudo mkdir -p /etc/kubernetes/manifests
    sudo wget -N -P /etc/kubernetes/manifests https://raw.githubusercontent.com/projectcalico/calico-cni/k8s-1.1-docs/samples/kubernetes/master/kubernetes-master.manifest
    ```

4.  Check the progress by running `docker ps`.  After a while, you should see the `etcd`, `apiserver`, `controller-manager`, `scheduler`, and `kube-proxy` containers running.

    > Note: it may take some time for all the containers to start. Don't worry if `docker ps` doesn't show any containers for a while or if some containers start before others.

### Install Calico's etcd on the master

Calico needs its own etcd cluster to store its state.  In this guide we install a single-node cluster on the master server.

> Note: In a production deployment we recommend running a distributed etcd cluster for redundancy. In this guide, we use a single etcd for simplicitly.

1.  Download the template manifest file:

    ```
    wget https://raw.githubusercontent.com/projectcalico/calico-cni/k8s-1.1-docs/samples/kubernetes/master/calico-etcd.manifest
    ```

2.  Replace all instances of `<MASTER_IPV4>` in the `calico-etcd.manifest` file with your master's IP address.

3.  Then, move the file to the `/etc/kubernetes/manifests` directory:

    ```
    sudo mv -f calico-etcd.manifest /etc/kubernetes/manifests
    ```

### Install Calico on the master

We need to install Calico on the master.  This allows the master to route packets to the pods on other nodes.

1.  Install the `calicoctl` tool:

    ```
    wget https://github.com/projectcalico/calico-containers/releases/download/v0.15.0/calicoctl
    chmod +x calicoctl
    sudo mv calicoctl /usr/bin
    ```

2.  Prefetch the calico/node container (this ensures that the Calico service starts immediately when we enable it):

    ```
    sudo docker pull calico/node:v0.15.0
    ```

3.  Download the `network-environment` template from the `calico-kubernetes` repository:

    ```
    wget -O network-environment https://raw.githubusercontent.com/projectcalico/calico-cni/k8s-1.1-docs/samples/kubernetes/master/network-environment-template
    ```

4.  Edit `network-environment` to represent this node's settings:

    -   Replace `<KUBERNETES_MASTER>` with the IP address of the master.  This should be the source IP address used to reach the Kubernetes worker nodes.

5.  Move `network-environment` into `/etc`:

    ```
    sudo mv -f network-environment /etc
    ```

6.  Install, enable, and start the `calico-node` service:

    ```
    sudo wget -N -P /etc/systemd https://raw.githubusercontent.com/projectcalico/calico-cni/k8s-1.1-docs/samples/kubernetes/common/calico-node.service
    sudo systemctl enable /etc/systemd/calico-node.service
    sudo systemctl start calico-node.service
    ```

## Set up the nodes

The following steps should be run on each Kubernetes node.

### Configure TLS

Worker nodes require three keys: `ca.pem`, `worker.pem`, and `worker-key.pem`.  We've already generated
`ca.pem` for use on the Master.  The worker public/private keypair should be generated for each Kubernetes node.

1.  Create the file `worker-openssl.cnf` with the following contents.

    ```
    [req]
    req_extensions = v3_req
    distinguished_name = req_distinguished_name
    [req_distinguished_name]
    [ v3_req ]
    basicConstraints = CA:FALSE
    keyUsage = nonRepudiation, digitalSignature, keyEncipherment
    subjectAltName = @alt_names
    [alt_names]
    IP.1 = $ENV::WORKER_IP
    ```

2.  Generate the necessary TLS assets for this worker. This relies on the worker's IP address, and the `ca.pem` file generated earlier in the guide.

    ```
    # Export this worker's IP address.
    export WORKER_IP=<WORKER_IPV4>
    ```

    ```
    # Generate keys.
    openssl genrsa -out worker-key.pem 2048
    openssl req -new -key worker-key.pem -out worker.csr -subj "/CN=worker-key" -config worker-openssl.cnf
    openssl x509 -req -in worker.csr -CA ca.pem -CAkey ca-key.pem -CAcreateserial -out worker.pem -days 365 -extensions v3_req -extfile worker-openssl.cnf
    ```

3.  Send the three files (`ca.pem`, `worker.pem`, and `worker-key.pem`) to the host (using scp, for example).

4.  Move the files to the `/etc/kubernetes/ssl` folder with the appropriate permissions:

    ```
    # Move keys
    sudo mkdir -p /etc/kubernetes/ssl/
    sudo mv -t /etc/kubernetes/ssl/ ca.pem worker.pem worker-key.pem

    # Set permissions
    sudo chmod 600 /etc/kubernetes/ssl/worker-key.pem
    sudo chown root:root /etc/kubernetes/ssl/worker-key.pem
    ```

### Configure the kubelet worker

1.  With your certs in place, create a kubeconfig for worker authentication in `/etc/kubernetes/worker-kubeconfig.yaml`; replace `<KUBERNETES_MASTER>` with the IP address of the master:

    ```
    apiVersion: v1
    kind: Config
    clusters:
    - name: local
      cluster:
        server: https://<KUBERNETES_MASTER>:443
        certificate-authority: /etc/kubernetes/ssl/ca.pem
    users:
    - name: kubelet
      user:
        client-certificate: /etc/kubernetes/ssl/worker.pem
        client-key: /etc/kubernetes/ssl/worker-key.pem
    contexts:
    - context:
        cluster: local
        user: kubelet
      name: kubelet-context
    current-context: kubelet-context
    ```

### Install Calico on the node

On your compute nodes, it is important that you install Calico before Kubernetes. We'll install Calico using the provided `calico-node.service` systemd unit file:

1.  Install the `calicoctl` binary:

    ```
    wget https://github.com/projectcalico/calico-containers/releases/download/v0.15.0/calicoctl
    chmod +x calicoctl
    sudo mv calicoctl /usr/bin
    ```

2.  Fetch the calico/node container:

    ```
    sudo docker pull calico/node:v0.15.0
    ```

3.  Download the `network-environment` template from the `calico-cni` repository:

    ```
    wget -O network-environment https://raw.githubusercontent.com/projectcalico/calico-cni/k8s-1.1-docs/samples/kubernetes/node/network-environment-template
    ```

4.  Edit `network-environment` to represent this node's settings:

    -   Replace `<DEFAULT_IPV4>` with the IP address of the node.
    -   Replace `<KUBERNETES_MASTER>` with the IP or hostname of the master.

5.  Move `network-environment` into `/etc`:

    ```
    sudo mv -f network-environment /etc
    ```

6.  Install the `calico-node` service:

    ```
    sudo wget -N -P /etc/systemd https://raw.githubusercontent.com/projectcalico/calico-cni/k8s-1.1-docs/samples/kubernetes/common/calico-node.service
    sudo systemctl enable /etc/systemd/calico-node.service
    sudo systemctl start calico-node.service
    ```

7.  Install the Calico CNI plugins:

    ```
    sudo mkdir -p /opt/cni/bin/
    sudo wget -N -P /opt/cni/bin/ https://github.com/projectcalico/calico-cni/releases/download/v1.0.0/calico
    sudo wget -N -P /opt/cni/bin/ https://github.com/projectcalico/calico-cni/releases/download/v1.0.0/calico-ipam
    sudo chmod +x /opt/cni/bin/calico /opt/cni/bin/calico-ipam
    ```

8.  Create a CNI network configuration file, which tells Kubernetes to create a network named `calico-k8s-network` and to use the calico plugins for that network.  Create file `/etc/cni/net.d/10-calico.conf` with the following contents, replacing `<KUBERNETES_MASTER>` with the IP of the master (this file should be the same on each node):

    ```
    # Make the directory structure.
    mkdir -p /etc/cni/net.d

    # Make the network configuration file
    cat >/etc/rkt/net.d/10-calico.conf <<EOF
    {
        "name": "calico-k8s-network",
        "type": "calico",
        "etcd_authority": "<KUBERNETES_MASTER>:6666",
        "log_level": "info",
        "ipam": {
            "type": "calico-ipam"
        }
    }
    EOF
    ```

    Since this is the only network we create, it will be used by default by the kubelet.

9.  Verify that Calico started correctly:

    ```
    calicoctl status
    ```

    should show that Felix (Calico's per-node agent) is running and the there should be a BGP status line for each other node that you've configured and the master.  The "Info" column should show "Established":

    ```
    $ calicoctl status
    calico-node container is running. Status: Up 15 hours
    Running felix version 1.3.0rc5
    
    IPv4 BGP status
    +---------------+-------------------+-------+----------+-------------+
    |  Peer address |     Peer type     | State |  Since   |     Info    |
    +---------------+-------------------+-------+----------+-------------+
    | 172.18.203.41 | node-to-node mesh |   up  | 17:32:26 | Established |
    | 172.18.203.42 | node-to-node mesh |   up  | 17:32:25 | Established |
    +---------------+-------------------+-------+----------+-------------+
    
    IPv6 BGP status
    +--------------+-----------+-------+-------+------+
    | Peer address | Peer type | State | Since | Info |
    +--------------+-----------+-------+-------+------+
    +--------------+-----------+-------+-------+------+
    ```

    If the "Info" column shows "Active" or some other value then Calico is having difficulty connecting to the other host.  Check the IP address of the peer is correct and check that Calico is using the correct local IP address (set in the `network-environment` file above).

### Install Kubernetes on the Node

1.  Download and Install the kubelet binary:

    ```
    sudo wget -N -P /usr/bin http://storage.googleapis.com/kubernetes-release/release/v1.1.4/bin/linux/amd64/kubelet
    sudo chmod +x /usr/bin/kubelet
    ```

2.  Install the `kubelet` systemd unit file:

    ```
    # Download the unit file.
    sudo wget -N -P /etc/systemd  https://raw.githubusercontent.com/projectcalico/calico-cni/k8s-1.1-docs/samples/kubernetes/node/kubelet.service

    # Enable and start the unit files so that they run on boot
    sudo systemctl enable /etc/systemd/kubelet.service
    sudo systemctl start kubelet.service
    ```

3.  Download the `kube-proxy` manifest:

    ```
    wget https://raw.githubusercontent.com/projectcalico/calico-cni/k8s-1.1-docs/samples/kubernetes/node/kube-proxy.manifest
    ```

4.  In that file, replace `<KUBERNETES_MASTER>` with your master's IP. Then move it into place:

    ```
    sudo mkdir -p /etc/kubernetes/manifests/
    sudo mv kube-proxy.manifest /etc/kubernetes/manifests/
    ```

## Configure kubectl remote access

To administer your cluster from a separate host (e.g your laptop), you will need the root CA generated earlier, as well as an admin public/private keypair (`ca.pem`, `admin.pem`, `admin-key.pem`). Run the following steps on the machine which you will use to control your cluster.

1. Download the kubectl binary.

   ```
   sudo wget -N -P /usr/bin http://storage.googleapis.com/kubernetes-release/release/v1.1.4/bin/linux/amd64/kubectl
   sudo chmod +x /usr/bin/kubectl
   ```

2. Generate the admin public/private keypair.

3. Export the necessary variables, substituting in correct values for your machine.

   ```
   # Export the appropriate paths.
   export CA_CERT_PATH=<PATH_TO_CA_PEM>
   export ADMIN_CERT_PATH=<PATH_TO_ADMIN_PEM>
   export ADMIN_KEY_PATH=<PATH_TO_ADMIN_KEY_PEM>

   # Export the Master's IP address.
   export MASTER_IPV4=<MASTER_IPV4>
   ```

4. Configure your host `kubectl` with the admin credentials:

   ```
   kubectl config set-cluster calico-cluster --server=https://${MASTER_IPV4} --certificate-authority=${CA_CERT_PATH}
   kubectl config set-credentials calico-admin --certificate-authority=${CA_CERT_PATH} --client-key=${ADMIN_KEY_PATH} --client-certificate=${ADMIN_CERT_PATH}
   kubectl config set-context calico --cluster=calico-cluster --user=calico-admin
   kubectl config use-context calico
   ```

Check your work with `kubectl get nodes`, which should succeed and display the nodes.

## Install the DNS Addon

Most Kubernetes deployments will require the DNS addon for service discovery. To install DNS, create the skydns service and replication controller provided.  This step makes use of the kubectl configuration made above.

```
kubectl create -f https://raw.githubusercontent.com/projectcalico/calico-cni/k8s-1.1-docs/samples/kubernetes/master/dns/skydns.yaml
```

## Install the Kubernetes UI Addon (Optional)

The Kubernetes UI can be installed using `kubectl` to run the following manifest file.

```
kubectl create -f https://raw.githubusercontent.com/projectcalico/calico-cni/k8s-1.1-docs/samples/kubernetes/master/kube-ui/kube-ui.yaml
```

## Launch other Services With Calico-Kubernetes

At this point, you have a fully functioning cluster running on Kubernetes with a master and two nodes networked with Calico. You can now follow any of the [standard documentation](../../examples/) to set up other services on your cluster.

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

[![Analytics](https://ga-beacon.appspot.com/UA-52125893-3/kubernetes/docs/getting-started-guides/ubuntu_calico.md?pixel)](https://github.com/igrigorik/ga-beacon)

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/ubuntu-calico.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
