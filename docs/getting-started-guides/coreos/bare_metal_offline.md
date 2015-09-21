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

<strong>
The latest 1.0.x release of this document can be found
[here](http://releases.k8s.io/release-1.0/docs/getting-started-guides/coreos/bare_metal_offline.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
Bare Metal CoreOS with Kubernetes (OFFLINE)
------------------------------------------
Deploy a CoreOS running Kubernetes environment. This particular guild is made to help those in an OFFLINE system, wither for testing a POC before the real deal, or you are restricted to be totally offline for your applications.

**Table of Contents**

- [Prerequisites](#prerequisites)
- [High Level Design](#high-level-design)
- [This Guides variables](#this-guides-variables)
- [Setup PXELINUX CentOS](#setup-pxelinux-centos)
- [Adding CoreOS to PXE](#adding-coreos-to-pxe)
- [DHCP configuration](#dhcp-configuration)
- [Kubernetes](#kubernetes)
- [Cloud Configs](#cloud-configs)
    - [master.yml](#masteryml)
    - [node.yml](#nodeyml)
- [New pxelinux.cfg file](#new-pxelinuxcfg-file)
- [Specify the pxelinux targets](#specify-the-pxelinux-targets)
- [Creating test pod](#creating-test-pod)
- [Helping commands for debugging](#helping-commands-for-debugging)


## Prerequisites

1. Installed *CentOS 6* for PXE server
2. At least two bare metal nodes to work with

## High Level Design

1. Manage the tftp directory
  * /tftpboot/(coreos)(centos)(RHEL)
  * /tftpboot/pxelinux.0/(MAC) -> linked to Linux image config file
2. Update per install the link for pxelinux
3. Update the DHCP config to reflect the host needing deployment
4. Setup nodes to deploy CoreOS creating a etcd cluster.
5. Have no access to the public [etcd discovery tool](https://discovery.etcd.io/).
6. Installing the CoreOS slaves to become Kubernetes nodes.

## This Guides variables

| Node Description              | MAC               | IP          |
| :---------------------------- | :---------------: | :---------: |
| CoreOS/etcd/Kubernetes Master | d0:00:67:13:0d:00 | 10.20.30.40 |
| CoreOS Slave 1                | d0:00:67:13:0d:01 | 10.20.30.41 |
| CoreOS Slave 2                | d0:00:67:13:0d:02 | 10.20.30.42 |


## Setup PXELINUX CentOS

To setup CentOS PXELINUX environment there is a complete [guide here](http://docs.fedoraproject.org/en-US/Fedora/7/html/Installation_Guide/ap-pxe-server.html). This section is the abbreviated version.

1. Install packages needed on CentOS

        sudo yum install tftp-server dhcp syslinux

2. `vi /etc/xinetd.d/tftp` to enable tftp service and change disable to 'no'
        disable = no

3. Copy over the syslinux images we will need.

        su -
        mkdir -p /tftpboot
        cd /tftpboot
        cp /usr/share/syslinux/pxelinux.0 /tftpboot
        cp /usr/share/syslinux/menu.c32 /tftpboot
        cp /usr/share/syslinux/memdisk /tftpboot
        cp /usr/share/syslinux/mboot.c32 /tftpboot
        cp /usr/share/syslinux/chain.c32 /tftpboot

        /sbin/service dhcpd start
        /sbin/service xinetd start
        /sbin/chkconfig tftp on

4. Setup default boot menu

        mkdir /tftpboot/pxelinux.cfg
        touch /tftpboot/pxelinux.cfg/default

5. Edit the menu `vi /tftpboot/pxelinux.cfg/default`

        default menu.c32
        prompt 0
        timeout 15
        ONTIMEOUT local
        display boot.msg

        MENU TITLE Main Menu

        LABEL local
                MENU LABEL Boot local hard drive
                LOCALBOOT 0

Now you should have a working PXELINUX setup to image CoreOS nodes. You can verify the services by using VirtualBox locally or with bare metal servers.

## Adding CoreOS to PXE

This section describes how to setup the CoreOS images to live alongside a pre-existing PXELINUX environment.

1. Find or create the TFTP root directory that everything will be based off of.
    * For this document we will assume `/tftpboot/` is our root directory.
2. Once we know and have our tftp root directory we will create a new directory structure for our CoreOS images.
3. Download the CoreOS PXE files provided by the CoreOS team.

        MY_TFTPROOT_DIR=/tftpboot
        mkdir -p $MY_TFTPROOT_DIR/images/coreos/
        cd $MY_TFTPROOT_DIR/images/coreos/
        wget http://stable.release.core-os.net/amd64-usr/current/coreos_production_pxe.vmlinuz
        wget http://stable.release.core-os.net/amd64-usr/current/coreos_production_pxe.vmlinuz.sig
        wget http://stable.release.core-os.net/amd64-usr/current/coreos_production_pxe_image.cpio.gz
        wget http://stable.release.core-os.net/amd64-usr/current/coreos_production_pxe_image.cpio.gz.sig
        gpg --verify coreos_production_pxe.vmlinuz.sig
        gpg --verify coreos_production_pxe_image.cpio.gz.sig

4. Edit the menu `vi /tftpboot/pxelinux.cfg/default` again

        default menu.c32
        prompt 0
        timeout 300
        ONTIMEOUT local
        display boot.msg

        MENU TITLE Main Menu

        LABEL local
                MENU LABEL Boot local hard drive
                LOCALBOOT 0

        MENU BEGIN CoreOS Menu

            LABEL coreos-master
                MENU LABEL CoreOS Master
                KERNEL images/coreos/coreos_production_pxe.vmlinuz
                APPEND initrd=images/coreos/coreos_production_pxe_image.cpio.gz cloud-config-url=http://<xxx.xxx.xxx.xxx>/pxe-cloud-config-single-master.yml

            LABEL coreos-slave
                MENU LABEL CoreOS Slave
                KERNEL images/coreos/coreos_production_pxe.vmlinuz
                APPEND initrd=images/coreos/coreos_production_pxe_image.cpio.gz cloud-config-url=http://<xxx.xxx.xxx.xxx>/pxe-cloud-config-slave.yml
        MENU END

This configuration file will now boot from local drive but have the option to PXE image CoreOS.

## DHCP configuration

This section covers configuring the DHCP server to hand out our new images. In this case we are assuming that there are other servers that will boot alongside other images.

1. Add the `filename` to the _host_ or _subnet_ sections.

        filename "/tftpboot/pxelinux.0";

2. At this point we want to make pxelinux configuration files that will be the templates for the different CoreOS deployments.

        subnet 10.20.30.0 netmask 255.255.255.0 {
                next-server 10.20.30.242;
                option broadcast-address 10.20.30.255;
                filename "<other default image>";

                ...
                # http://www.syslinux.org/wiki/index.php/PXELINUX
                host core_os_master {
                        hardware ethernet d0:00:67:13:0d:00;
                        option routers 10.20.30.1;
                        fixed-address 10.20.30.40;
                        option domain-name-servers 10.20.30.242;
                        filename "/pxelinux.0";
                }
                host core_os_slave {
                        hardware ethernet d0:00:67:13:0d:01;
                        option routers 10.20.30.1;
                        fixed-address 10.20.30.41;
                        option domain-name-servers 10.20.30.242;
                        filename "/pxelinux.0";
                }
                host core_os_slave2 {
                        hardware ethernet d0:00:67:13:0d:02;
                        option routers 10.20.30.1;
                        fixed-address 10.20.30.42;
                        option domain-name-servers 10.20.30.242;
                        filename "/pxelinux.0";
                }
                ...
        }

We will be specifying the node configuration later in the guide.

## Kubernetes

To deploy our configuration we need to create an `etcd` master. To do so we want to pxe CoreOS with a specific cloud-config.yml. There are two options we have here.
1. Is to template the cloud config file and programmatically create new static configs for different cluster setups.
2. Have a service discovery protocol running in our stack to do auto discovery.

This demo we just make a static single `etcd` server to host our Kubernetes and `etcd` master servers.

Since we are OFFLINE here most of the helping processes in CoreOS and Kubernetes are then limited. To do our setup we will then have to download and serve up our binaries for Kubernetes in our local environment.

An easy solution is to host a small web server on the DHCP/TFTP host for all our binaries to make them available to the local CoreOS PXE machines.

To get this up and running we are going to setup a simple `apache` server to serve our binaries needed to bootstrap Kubernetes.

This is on the PXE server from the previous section:

    rm /etc/httpd/conf.d/welcome.conf
    cd /var/www/html/
    wget -O kube-register  https://github.com/kelseyhightower/kube-register/releases/download/v0.0.2/kube-register-0.0.2-linux-amd64
    wget -O setup-network-environment https://github.com/kelseyhightower/setup-network-environment/releases/download/v1.0.0/setup-network-environment
    wget https://storage.googleapis.com/kubernetes-release/release/v0.15.0/bin/linux/amd64/kubernetes --no-check-certificate
    wget https://storage.googleapis.com/kubernetes-release/release/v0.15.0/bin/linux/amd64/kube-apiserver --no-check-certificate
    wget https://storage.googleapis.com/kubernetes-release/release/v0.15.0/bin/linux/amd64/kube-controller-manager --no-check-certificate
    wget https://storage.googleapis.com/kubernetes-release/release/v0.15.0/bin/linux/amd64/kube-scheduler --no-check-certificate
    wget https://storage.googleapis.com/kubernetes-release/release/v0.15.0/bin/linux/amd64/kubectl --no-check-certificate
    wget https://storage.googleapis.com/kubernetes-release/release/v0.15.0/bin/linux/amd64/kubecfg --no-check-certificate
    wget https://storage.googleapis.com/kubernetes-release/release/v0.15.0/bin/linux/amd64/kubelet --no-check-certificate
    wget https://storage.googleapis.com/kubernetes-release/release/v0.15.0/bin/linux/amd64/kube-proxy --no-check-certificate
    wget -O flanneld https://storage.googleapis.com/k8s/flanneld --no-check-certificate

This sets up our binaries we need to run Kubernetes. This would need to be enhanced to download from the Internet for updates in the future.

Now for the good stuff!

## Cloud Configs

The following config files are tailored for the OFFLINE version of a Kubernetes deployment.

These are based on the work found here: [master.yml](cloud-configs/master.yaml), [node.yml](cloud-configs/node.yaml)

To make the setup work, you need to replace a few placeholders:

 - Replace `<PXE_SERVER_IP>` with your PXE server ip address (e.g. 10.20.30.242)
 - Replace `<MASTER_SERVER_IP>` with the Kubernetes master ip address (e.g. 10.20.30.40)
 - If you run a private docker registry, replace `rdocker.example.com` with your docker registry dns name.
 - If you use a proxy, replace `rproxy.example.com` with your proxy server (and port)
 - Add your own SSH public key(s) to the cloud config at the end

### master.yml

On the PXE server make and fill in the variables `vi /var/www/html/coreos/pxe-cloud-config-master.yml`.


    #cloud-config
    ---
    write_files:
      - path: /opt/bin/waiter.sh
        owner: root
        content: |
          #! /usr/bin/bash
          until curl http://127.0.0.1:4001/v2/machines; do sleep 2; done
      - path: /opt/bin/kubernetes-download.sh
        owner: root
        permissions: 0755
        content: |
          #! /usr/bin/bash
          /usr/bin/wget -N -P "/opt/bin" "http://<PXE_SERVER_IP>/kubectl"
          /usr/bin/wget -N -P "/opt/bin" "http://<PXE_SERVER_IP>/kubernetes"
          /usr/bin/wget -N -P "/opt/bin" "http://<PXE_SERVER_IP>/kubecfg"
          chmod +x /opt/bin/*
      - path: /etc/profile.d/opt-path.sh
        owner: root
        permissions: 0755
        content: |
          #! /usr/bin/bash
          PATH=$PATH/opt/bin
    coreos:
      units:
        - name: 10-eno1.network
          runtime: true
          content: |
            [Match]
            Name=eno1
            [Network]
            DHCP=yes
        - name: 20-nodhcp.network
          runtime: true
          content: |
            [Match]
            Name=en*
            [Network]
            DHCP=none
        - name: get-kube-tools.service
          runtime: true
          command: start
          content: |
            [Service]
            ExecStartPre=-/usr/bin/mkdir -p /opt/bin
            ExecStart=/opt/bin/kubernetes-download.sh
            RemainAfterExit=yes
            Type=oneshot
        - name: setup-network-environment.service
          command: start
          content: |
            [Unit]
            Description=Setup Network Environment
            Documentation=https://github.com/kelseyhightower/setup-network-environment
            Requires=network-online.target
            After=network-online.target
            [Service]
            ExecStartPre=-/usr/bin/mkdir -p /opt/bin
            ExecStartPre=/usr/bin/wget -N -P /opt/bin http://<PXE_SERVER_IP>/setup-network-environment
            ExecStartPre=/usr/bin/chmod +x /opt/bin/setup-network-environment
            ExecStart=/opt/bin/setup-network-environment
            RemainAfterExit=yes
            Type=oneshot
        - name: etcd.service
          command: start
          content: |
            [Unit]
            Description=etcd
            Requires=setup-network-environment.service
            After=setup-network-environment.service
            [Service]
            EnvironmentFile=/etc/network-environment
            User=etcd
            PermissionsStartOnly=true
            ExecStart=/usr/bin/etcd \
            --name ${DEFAULT_IPV4} \
            --addr ${DEFAULT_IPV4}:4001 \
            --bind-addr 0.0.0.0 \
            --cluster-active-size 1 \
            --data-dir /var/lib/etcd \
            --http-read-timeout 86400 \
            --peer-addr ${DEFAULT_IPV4}:7001 \
            --snapshot true
            Restart=always
            RestartSec=10s
        - name: fleet.socket
          command: start
          content: |
            [Socket]
            ListenStream=/var/run/fleet.sock
        - name: fleet.service
          command: start
          content: |
            [Unit]
            Description=fleet daemon
            Wants=etcd.service
            After=etcd.service
            Wants=fleet.socket
            After=fleet.socket
            [Service]
            Environment="FLEET_ETCD_SERVERS=http://127.0.0.1:4001"
            Environment="FLEET_METADATA=role=master"
            ExecStart=/usr/bin/fleetd
            Restart=always
            RestartSec=10s
        - name: etcd-waiter.service
          command: start
          content: |
            [Unit]
            Description=etcd waiter
            Wants=network-online.target
            Wants=etcd.service
            After=etcd.service
            After=network-online.target
            Before=flannel.service
            Before=setup-network-environment.service
            [Service]
            ExecStartPre=/usr/bin/chmod +x /opt/bin/waiter.sh
            ExecStart=/usr/bin/bash /opt/bin/waiter.sh
            RemainAfterExit=true
            Type=oneshot
        - name: flannel.service
          command: start
          content: |
            [Unit]
            Wants=etcd-waiter.service
            After=etcd-waiter.service
            Requires=etcd.service
            After=etcd.service
            After=network-online.target
            Wants=network-online.target
            Description=flannel is an etcd backed overlay network for containers
            [Service]
            Type=notify
            ExecStartPre=-/usr/bin/mkdir -p /opt/bin
            ExecStartPre=/usr/bin/wget -N -P /opt/bin http://<PXE_SERVER_IP>/flanneld
            ExecStartPre=/usr/bin/chmod +x /opt/bin/flanneld
            ExecStartPre=-/usr/bin/etcdctl mk /coreos.com/network/config '{"Network":"10.100.0.0/16", "Backend": {"Type": "vxlan"}}'
            ExecStart=/opt/bin/flanneld
        - name: kube-apiserver.service
          command: start
          content: |
            [Unit]
            Description=Kubernetes API Server
            Documentation=https://github.com/kubernetes/kubernetes
            Requires=etcd.service
            After=etcd.service
            [Service]
            ExecStartPre=-/usr/bin/mkdir -p /opt/bin
            ExecStartPre=/usr/bin/wget -N -P /opt/bin http://<PXE_SERVER_IP>/kube-apiserver
            ExecStartPre=/usr/bin/chmod +x /opt/bin/kube-apiserver
            ExecStart=/opt/bin/kube-apiserver \
            --address=0.0.0.0 \
            --port=8080 \
            --service-cluster-ip-range=10.100.0.0/16 \
            --etcd-servers=http://127.0.0.1:4001 \
            --logtostderr=true
            Restart=always
            RestartSec=10
        - name: kube-controller-manager.service
          command: start
          content: |
            [Unit]
            Description=Kubernetes Controller Manager
            Documentation=https://github.com/kubernetes/kubernetes
            Requires=kube-apiserver.service
            After=kube-apiserver.service
            [Service]
            ExecStartPre=/usr/bin/wget -N -P /opt/bin http://<PXE_SERVER_IP>/kube-controller-manager
            ExecStartPre=/usr/bin/chmod +x /opt/bin/kube-controller-manager
            ExecStart=/opt/bin/kube-controller-manager \
            --master=127.0.0.1:8080 \
            --logtostderr=true
            Restart=always
            RestartSec=10
        - name: kube-scheduler.service
          command: start
          content: |
            [Unit]
            Description=Kubernetes Scheduler
            Documentation=https://github.com/kubernetes/kubernetes
            Requires=kube-apiserver.service
            After=kube-apiserver.service
            [Service]
            ExecStartPre=/usr/bin/wget -N -P /opt/bin http://<PXE_SERVER_IP>/kube-scheduler
            ExecStartPre=/usr/bin/chmod +x /opt/bin/kube-scheduler
            ExecStart=/opt/bin/kube-scheduler --master=127.0.0.1:8080
            Restart=always
            RestartSec=10
        - name: kube-register.service
          command: start
          content: |
            [Unit]
            Description=Kubernetes Registration Service
            Documentation=https://github.com/kelseyhightower/kube-register
            Requires=kube-apiserver.service
            After=kube-apiserver.service
            Requires=fleet.service
            After=fleet.service
            [Service]
            ExecStartPre=/usr/bin/wget -N -P /opt/bin http://<PXE_SERVER_IP>/kube-register
            ExecStartPre=/usr/bin/chmod +x /opt/bin/kube-register
            ExecStart=/opt/bin/kube-register \
            --metadata=role=node \
            --fleet-endpoint=unix:///var/run/fleet.sock \
            --healthz-port=10248 \
            --api-endpoint=http://127.0.0.1:8080
            Restart=always
            RestartSec=10
      update:
        group: stable
        reboot-strategy: off
    ssh_authorized_keys:
      - ssh-rsa AAAAB3NzaC1yc2EAAAAD...


### node.yml

On the PXE server make and fill in the variables `vi /var/www/html/coreos/pxe-cloud-config-slave.yml`.

    #cloud-config
    ---
    write_files:
      - path: /etc/default/docker
        content: |
          DOCKER_EXTRA_OPTS='--insecure-registry="rdocker.example.com:5000"'
    coreos:
      units:
        - name: 10-eno1.network
          runtime: true
          content: |
            [Match]
            Name=eno1
            [Network]
            DHCP=yes
        - name: 20-nodhcp.network
          runtime: true
          content: |
            [Match]
            Name=en*
            [Network]
            DHCP=none
        - name: etcd.service
          mask: true
        - name: docker.service
          drop-ins:
            - name: 50-insecure-registry.conf
              content: |
                [Service]
                Environment="HTTP_PROXY=http://rproxy.example.com:3128/" "NO_PROXY=localhost,127.0.0.0/8,rdocker.example.com"
        - name: fleet.service
          command: start
          content: |
            [Unit]
            Description=fleet daemon
            Wants=fleet.socket
            After=fleet.socket
            [Service]
            Environment="FLEET_ETCD_SERVERS=http://<MASTER_SERVER_IP>:4001"
            Environment="FLEET_METADATA=role=node"
            ExecStart=/usr/bin/fleetd
            Restart=always
            RestartSec=10s
        - name: flannel.service
          command: start
          content: |
            [Unit]
            After=network-online.target
            Wants=network-online.target
            Description=flannel is an etcd backed overlay network for containers
            [Service]
            Type=notify
            ExecStartPre=-/usr/bin/mkdir -p /opt/bin
            ExecStartPre=/usr/bin/wget -N -P /opt/bin http://<PXE_SERVER_IP>/flanneld
            ExecStartPre=/usr/bin/chmod +x /opt/bin/flanneld
            ExecStart=/opt/bin/flanneld -etcd-endpoints http://<MASTER_SERVER_IP>:4001
        - name: docker.service
          command: start
          content: |
            [Unit]
            After=flannel.service
            Wants=flannel.service
            Description=Docker Application Container Engine
            Documentation=http://docs.docker.io
            [Service]
            EnvironmentFile=-/etc/default/docker
            EnvironmentFile=/run/flannel/subnet.env
            ExecStartPre=/bin/mount --make-rprivate /
            ExecStart=/usr/bin/docker -d --bip=${FLANNEL_SUBNET} --mtu=${FLANNEL_MTU} -s=overlay -H fd:// ${DOCKER_EXTRA_OPTS}
            [Install]
            WantedBy=multi-user.target
        - name: setup-network-environment.service
          command: start
          content: |
            [Unit]
            Description=Setup Network Environment
            Documentation=https://github.com/kelseyhightower/setup-network-environment
            Requires=network-online.target
            After=network-online.target
            [Service]
            ExecStartPre=-/usr/bin/mkdir -p /opt/bin
            ExecStartPre=/usr/bin/wget -N -P /opt/bin http://<PXE_SERVER_IP>/setup-network-environment
            ExecStartPre=/usr/bin/chmod +x /opt/bin/setup-network-environment
            ExecStart=/opt/bin/setup-network-environment
            RemainAfterExit=yes
            Type=oneshot
        - name: kube-proxy.service
          command: start
          content: |
            [Unit]
            Description=Kubernetes Proxy
            Documentation=https://github.com/kubernetes/kubernetes
            Requires=setup-network-environment.service
            After=setup-network-environment.service
            [Service]
            ExecStartPre=/usr/bin/wget -N -P /opt/bin http://<PXE_SERVER_IP>/kube-proxy
            ExecStartPre=/usr/bin/chmod +x /opt/bin/kube-proxy
            ExecStart=/opt/bin/kube-proxy \
            --etcd-servers=http://<MASTER_SERVER_IP>:4001 \
            --logtostderr=true
            Restart=always
            RestartSec=10
        - name: kube-kubelet.service
          command: start
          content: |
            [Unit]
            Description=Kubernetes Kubelet
            Documentation=https://github.com/kubernetes/kubernetes
            Requires=setup-network-environment.service
            After=setup-network-environment.service
            [Service]
            EnvironmentFile=/etc/network-environment
            ExecStartPre=/usr/bin/wget -N -P /opt/bin http://<PXE_SERVER_IP>/kubelet
            ExecStartPre=/usr/bin/chmod +x /opt/bin/kubelet
            ExecStart=/opt/bin/kubelet \
            --address=0.0.0.0 \
            --port=10250 \
            --hostname-override=${DEFAULT_IPV4} \
            --api-servers=<MASTER_SERVER_IP>:8080 \
            --healthz-bind-address=0.0.0.0 \
            --healthz-port=10248 \
            --logtostderr=true
            Restart=always
            RestartSec=10
      update:
        group: stable
        reboot-strategy: off
    ssh_authorized_keys:
      - ssh-rsa AAAAB3NzaC1yc2EAAAAD...


## New pxelinux.cfg file

Create a pxelinux target file for a _slave_ node: `vi /tftpboot/pxelinux.cfg/coreos-node-slave`

    default coreos
    prompt 1
    timeout 15

    display boot.msg

    label coreos
      menu default
      kernel images/coreos/coreos_production_pxe.vmlinuz
      append initrd=images/coreos/coreos_production_pxe_image.cpio.gz cloud-config-url=http://<pxe-host-ip>/coreos/pxe-cloud-config-slave.yml console=tty0 console=ttyS0 coreos.autologin=tty1 coreos.autologin=ttyS0

And one for the _master_ node: `vi /tftpboot/pxelinux.cfg/coreos-node-master`

    default coreos
    prompt 1
    timeout 15

    display boot.msg

    label coreos
      menu default
      kernel images/coreos/coreos_production_pxe.vmlinuz
      append initrd=images/coreos/coreos_production_pxe_image.cpio.gz cloud-config-url=http://<pxe-host-ip>/coreos/pxe-cloud-config-master.yml console=tty0 console=ttyS0 coreos.autologin=tty1 coreos.autologin=ttyS0

## Specify the pxelinux targets

Now that we have our new targets setup for master and slave we want to configure the specific hosts to those targets. We will do this by using the pxelinux mechanism of setting a specific MAC addresses to a specific pxelinux.cfg file.

Refer to the MAC address table in the beginning of this guide. Documentation for more details can be found [here](http://www.syslinux.org/wiki/index.php/PXELINUX).

    cd /tftpboot/pxelinux.cfg
    ln -s coreos-node-master 01-d0-00-67-13-0d-00
    ln -s coreos-node-slave 01-d0-00-67-13-0d-01
    ln -s coreos-node-slave 01-d0-00-67-13-0d-02


Reboot these servers to get the images PXEd and ready for running containers!

## Creating test pod

Now that the CoreOS with Kubernetes installed is up and running lets spin up some Kubernetes pods to demonstrate the system.

See [a simple nginx example](../../../docs/user-guide/simple-nginx.md) to try out your new cluster.

For more complete applications, please look in the [examples directory](../../../examples/).

## Helping commands for debugging

List all keys in etcd:

     etcdctl ls --recursive

List fleet machines

    fleetctl list-machines

Check system status of services on master:

    systemctl status kube-apiserver
    systemctl status kube-controller-manager
    systemctl status kube-scheduler
    systemctl status kube-register

Check system status of services on a node:

    systemctl status kube-kubelet
    systemctl status docker.service

List Kubernetes

    kubectl get pods
    kubectl get nodes


Kill all pods:

    for i in `kubectl get pods | awk '{print $1}'`; do kubectl stop pod $i; done


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/coreos/bare_metal_offline.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
