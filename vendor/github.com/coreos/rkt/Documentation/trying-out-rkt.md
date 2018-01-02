# Trying out rkt

This document introduces the basics of getting rkt and running a container with it. For a more in-depth guide to building application containers and running them with rkt, check out the [getting started guide][getstart].

Giving rkt a spin takes just a few basic steps, detailed below.

1. Install rkt:
  * [CoreOS Linux][run-coreos] comes with rkt installed and configured. You can skip straight to [running a container with rkt][id-rkt-run-by-name].
  * On other Linux distributions, [grab the latest rkt binary][id-lin-rkt-bin], or [the distribution's rkt package][id-lin-rkt-pkg].
  * On Mac or Windows, you can [use a Vagrant virtual machine to run rkt][id-vagrant-rkt]
2. [Configure rkt][id-config-rkt]: Optional steps that make it simpler to experiment with rkt.
3. [Download][id-rkt-fetch] and [run a container with rkt][id-rkt-run].

## Using rkt on Linux

rkt is written in Go and can be compiled for several CPU architectures. The rkt project distributes binaries for amd64. These rkt binaries will run on any modern Linux amd64 kernel.

### Running the latest rkt binary

To start running the latest version of rkt on amd64, grab the release directly from the rkt GitHub project:

```
wget https://github.com/coreos/rkt/releases/download/v1.25.0/rkt-v1.25.0.tar.gz
tar xzvf rkt-v1.25.0.tar.gz
cd rkt-v1.25.0
./rkt help
```

### Installing rkt from a Linux distribution package

Another easy way to run rkt is to install it with your system's package manager, like *apt* on Debian or *dnf* on Fedora. Check for your Linux distribution in the [distributions list][distlist] to see if a rkt package is available.

## Running rkt in a Vagrant virtual machine

If your operating system isn't Linux, it's easy to run rkt in a Linux virtual machine with Vagrant. The instructions below start a virtual machine with rkt installed and ready to run.

### Vagrant on Mac and Windows

For Mac (and other Vagrant) users we have set up a `Vagrantfile`. Make sure you have [Vagrant][vagrant] 1.5.x or greater installed.

First, download the `Vagrantfile` and start a Linux machine with rkt installed by running `vagrant up`.

```
git clone https://github.com/coreos/rkt
cd rkt
vagrant up
```

### Vagrant on Linux

To use Vagrant on a Linux machine, you may want to use libvirt as a VMM instead of VirtualBox. To do so, install the necessary plugins, convert the box, and start the machine using the `libvirt` provider:

```
vagrant plugin install vagrant-libvirt
vagrant plugin install vagrant-mutate
vagrant mutate ubuntu/xenial64 libvirt
vagrant up --provider=libvirt
```

### Accessing the Vagrant VM and running rkt

With a subsequent `vagrant ssh` you will have access to run rkt:

```
vagrant ssh
rkt --help
```

Consult the rkt manual for more details:

```
man rkt
```

The Vagrant setup also includes bash-completion to assist with rkt subcommands and options.

#### Container networking on a Vagrant VM

To reach pods from your host, determine the IP address of the Vagrant machine:

```
vagrant ssh -c 'ip address'
...
3: enp0s8: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc pfifo_fast state UP group default qlen 1000
    link/ether 08:00:27:04:e4:5d brd ff:ff:ff:ff:ff:ff
    inet 172.28.128.3/24 brd 172.28.128.255 scope global enp0s8
       valid_lft forever preferred_lft forever
...
```

In this example, the Vagrant machine has the IP address `172.28.128.3`.

The following command starts an [`nginx`][docker-nginx] container, for simplicity using [*host networking*][host-network] to make the pod directly accessible on the host's network address and ports. Signature validation isn't supported for Docker registries and images, so `--insecure-options=image` switches off the signature check:

```
sudo rkt run --net=host --insecure-options=image docker://nginx
```

The nginx container is now accessible on the host under `http://172.28.128.3`.

In order to use containers with the default [*contained network*][contained-network], a route to the 172.16.28.0/24 container network must be configured from the host through the VM:

On Linux, execute:

```
sudo ip route add 172.16.28.0/24 via 172.28.128.3
```

On Mac OSX, execute:

```
sudo route -n add 172.16.28.0/24 172.28.128.3
```

Now nginx can be started using the default contained network:

```
$ sudo rkt run --insecure-options=image docker://nginx
$ rkt list
UUID		APP	IMAGE NAME					STATE	CREATED		STARTED		NETWORKS
0c3ab969	nginx	registry-1.docker.io/library/nginx:latest	running	2 minutes ago	2 minutes ago	default:ip4=172.16.28.2
```

In this example, the nginx container was assigned the IP address 172.16.28.2 (the address assigned on your system may vary). Since we established a route from the host to the `172.16.28.0/24` pod network the nginx container is now accessible on the host under `http://172.16.28.2`.

Success! The rest of the guide can now be followed normally.

## Configuring a rkt host

Once rkt is present on a machine, some optional configuration steps can make it easier to operate.

### SELinux

rkt supports running under SELinux mandatory access controls, but an SELinux policy needs to be tailored to your distribution. New rkt users on distributions other than CoreOS should temporarily [disable SELinux][disable-selinux] to make it easier to get started. If you can help package rkt for your distro, including SELinux policy support, [please lend a hand][distro-pkg-help]!

### Optional: Set up privilege separation

To allow different subcommands to use the least necessary privilege, rkt recognizes a `rkt` group that has read-write access to the rkt data directory. This allows [`rkt fetch`][rkt-fetch], which downloads and verifies images, to run as an unprivileged user who is a member of the `rkt` group.

If you skip this section, you can still run `sudo rkt fetch` instead, but setting up a `rkt` group is a good basic security practice for production use. The rkt repo includes a [`setup-data-dir.sh`][setup-data-dir] script that can help set up the appropriate permissions for unprivileged execution of subcommands that manipulate the local store, but not the execution environment:

```
sudo groupadd rkt
export WHOAMI=$(whoami); sudo gpasswd -a $WHOAMI rkt
sudo ./dist/scripts/setup-data-dir.sh
```

#### Trust the signing key to validate unprivileged fetches

Trust the signing key for etcd images. This step must be run as root because access to the keystore is restricted from even the `rkt` group:

```
sudo ./rkt trust --prefix coreos.com/etcd
```

#### Fetch an image as an unprivileged member of the rkt group

Test this out by retrieving an etcd image using a non-root user in the rkt group. Make sure your shell is restarted to enable the `rkt` group for your user, or
just run `newgrp rkt` to enable it and continue in the same session.

Now fetch the etcd image as an unprivileged user:

```
./rkt fetch coreos.com/etcd:v2.3.7
```

Success! Now rkt can fetch and download images as an unprivileged user.

## rkt basics

### Building an App Container Image

rkt's native image format is the App Container Image (ACI), defined in the [App Container spec][appc]. The [`acbuild`][acbuild] tool is a simple way to get started building ACIs. The [appc build repository][appc-build-repo] has resources for building ACIs from a number of popular applications.

The `docker2aci` tool [converts Docker images to ACIs][docker2aci], or rkt can [convert images directly from Docker registries on the fly][rktdocker].

The example below uses an [etcd][etcd] ACI constructed with `acbuild` by the etcd project's [`build-aci` script][build-aci].

### Downloading an ACI

rkt uses content addressable storage (CAS) to store an ACI on disk. In this example, an image is downloaded and added to the CAS. Downloading an image before running it is not strictly necessary &ndash; if an image is not present in the store, [rkt will attempt to retrieve it][aci-discovery] &ndash; but it illustrates how rkt works.

Since rkt verifies signatures by default, the first step is to [trust][signguide-establishing-trust] the [CoreOS public key][coreos-pubkey] used to sign the image, using the [`rkt trust`][rkt-trust] subcommand:

#### Trusting the signing key

```
$ sudo rkt trust --prefix=coreos.com/etcd
Prefix: "coreos.com/etcd"
Key: "https://coreos.com/dist/pubkeys/aci-pubkeys.gpg"
GPG key fingerprint is: 8B86 DE38 890D DB72 9186  7B02 5210 BD88 8818 2190
  CoreOS ACI Builder <release@coreos.com>
Are you sure you want to trust this key (yes/no)? yes
Trusting "https://coreos.com/dist/pubkeys/aci-pubkeys.gpg" for prefix "coreos.com/etcd".
Added key for prefix "coreos.com/etcd" at "/etc/rkt/trustedkeys/prefix.d/coreos.com/etcd/8b86de38890ddb7291867b025210bd8888182190"
```

For more information, see the [detailed, step-by-step guide for the signing procedure][signguide].

#### Fetching the ACI

Now that the CoreOS public key is trusted, fetch the ACI using [`rkt fetch`][rkt-fetch]. This step doesn't need root privileges if the rkt host has been [configured for privilege separation][id-privsep]:

```
$ rkt fetch coreos.com/etcd:v2.3.7
rkt: searching for app image coreos.com/etcd:v2.3.7
rkt: fetching image from https://github.com/coreos/etcd/releases/download/v2.3.7/etcd-v2.3.7-linux-amd64.aci
Downloading aci: [==========================================   ] 3.47 MB/3.7 MB
Downloading signature from https://github.com/coreos/etcd/releases/download/v2.3.7/etcd-v2.3.7-linux-amd64.aci.asc
rkt: signature verified:
  CoreOS ACI Builder <release@coreos.com>
sha512-7d28419b27d5ae56cca97f4c6ccdd309c...
```

#### Downloading images from private registries

Downloading container images from a private registry usually involves passing usernames and passwords or other kinds of credentials to the server. rkt supports different authentication regimes with configuration files. The [configuration documentation][configdoc] describes the file format and gives examples of setting up authentication with HTTP `basic` auth, OAuth bearer tokens, and other methods.

### The image in the local store

For the curious, it is possible to list the hash-identified files written to disk in rkt's CAS:

```
$ find /var/lib/rkt/cas/blob/
/var/lib/rkt/cas/blob/
/var/lib/rkt/cas/blob/sha512
/var/lib/rkt/cas/blob/sha512/1e
/var/lib/rkt/cas/blob/sha512/1e/sha512-7d28419b27d5ae56cca97f4c6ccdd309c95b967ca0119f6962b187d1287ec9967f49e367c36b0e44ecd73675bc06d112dec86386d0e9b84c2265cddd45d15020
```

According to the [App Container specification][aci-archives], the SHA-512 hash is that of the `tar` file compressed in the ACI, and can be examined with standard tools:

```
$ wget https://github.com/coreos/etcd/releases/download/v2.3.7/etcd-v2.3.7-linux-amd64.aci
...
$ gzip -dc etcd-v2.3.7-linux-amd64.aci > etcd-v2.3.7-linux-amd64.tar
$ sha512sum etcd-v2.3.7-linux-amd64.tar
7d28419b27d5ae56cca97f4c6ccdd309c95b967ca0119f6962b187d1287ec9967f49e367c36b0e44ecd73675bc06d112dec86386d0e9b84c2265cddd45d15020  etcd-v2.3.7-linux-amd64.tar
```

### Running an ACI with rkt

After it has been retrieved and stored locally, an ACI can be run by pointing [`rkt run`][rkt-run] at either the original image reference (in this case, `coreos.com/etcd:v2.3.7`), the ACI hash, or the full URL of the ACI. Therefore the following three examples are equivalent:

#### Running the container by ACI name and version

```
$ sudo rkt run coreos.com/etcd:v2.3.7
...
Press ^] three times to kill container
```

#### Running the container by ACI hash

```
$ sudo rkt run sha512-1eba37d9b344b33d272181e176da111e
...
^]]]
```

#### Running the container by ACI URL

```
$ sudo rkt run https://github.com/coreos/etcd/releases/download/v2.3.7/etcd-v2.3.7-linux-amd64.aci
...
^]]]
```

When given an ACI URL, `rkt` will do the appropriate ETag checking to fetch the latest version of the container image.

### Exiting rkt pods

As shown above, repeating the `^]` escape character three times kills the pod and detaches from its console to return to the user's shell.

The escape character `^]` is generated by `Ctrl-]` on a US keyboard. The required key combination will differ on other keyboard layouts. For example, the Swedish keyboard layout uses ```Ctrl-å``` on OS X, or ```Ctrl-^``` on Windows, to generate the ```^]``` escape character.


[acbuild]: https://github.com/containers/build
[aci-archives]: https://github.com/appc/spec/blob/master/spec/aci.md#image-archives
[aci-discovery]: https://github.com/appc/spec/blob/master/spec/discovery.md
[appc]: app-container.md
[appc-build-repo]: https://github.com/appc/build-repository
[build-aci]: https://github.com/coreos/etcd/blob/master/scripts/build-aci
[contained-network]: networking/overview.md#contained-mode
[coreos-pubkey]: https://coreos.com/dist/pubkeys/aci-pubkeys.gpg
[configdoc]: configuration.md
[disable-selinux]: https://www.centos.org/docs/5/html/5.1/Deployment_Guide/sec-sel-enable-disable.html
[distlist]: distributions.md
[distro-pkg-help]: https://github.com/coreos/rkt/issues?utf8=%E2%9C%93&q=is%3Aopen+is%3Aissue+label%3Aarea%2Fdistribution++label%3Adependency%2Fexternal
[docker2aci]: https://github.com/appc/docker2aci
[docker-nginx]: https://hub.docker.com/_/nginx/
[etcd]: https://coreos.com/etcd/
[getstart]: getting-started-guide.md
[host-network]: networking/overview.md#host-mode
[id-config-rkt]: #configuring-a-rkt-host
[id-lin-rkt-bin]: #running-the-latest-rkt-binary
[id-lin-rkt-pkg]: #installing-rkt-from-a-linux-distribution-package
[id-privsep]: #optional-set-up-privilege-separation
[id-rkt-fetch]: #downloading-an-aci
[id-rkt-run]: #running-an-aci-with-rkt
[id-rkt-run-by-name]: #running-the-container-by-aci-name-and-version
[id-vagrant-rkt]: #running-rkt-in-a-vagrant-virtual-machine
[rktdocker]: running-docker-images.md
[rkt-fetch]: subcommands/fetch.md
[rkt-run]: subcommands/run.md
[rkt-trust]: subcommands/trust.md
[run-coreos]: https://coreos.com/os/docs/latest/#running-coreos
[setup-data-dir]: https://github.com/coreos/rkt/blob/master/dist/scripts/setup-data-dir.sh
[signguide]: signing-and-verification-guide.md
[signguide-establishing-trust]: signing-and-verification-guide.md#establishing-trust
[vagrant]: https://www.vagrantup.com/
