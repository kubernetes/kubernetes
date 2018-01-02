# Installing rkt on popular Linux distributions

- [Arch](#arch)
- [CentOS](#centos)
- [CoreOS](#coreos)
- [Debian](#debian)
- [Fedora](#fedora)
- [NixOS](#nixos)
- [openSUSE](#opensuse)
- [Ubuntu](#ubuntu)
- [Void](#void)

## Upstream-maintained packages (manual installation from rkt project)
- [rpm-based](#rpm-based)
- [deb-based](#deb-based)


## Distribution-maintained packages (automatic installation from repositories)
If your distribution packages rkt, then you should generally use their version. However,
if you need a newer version, you may choose to manually install the rkt-provided rpm and deb packages.

## Arch

rkt is available in the [Community Repository][pkg-arch] and can be installed using pacman:
```
sudo pacman -S rkt
```

## CentOS

rkt is available in the [CentOS Community Build Service][pkg-centos] for CentOS 7.
However, this is [not yet ready for production use][rkt-1305] due to pending systemd upgrade issues.

## CoreOS

rkt is an integral part of CoreOS, installed with the operating system.
The [CoreOS releases page][coreos-releases] lists the version of rkt available in each CoreOS release channel.

If the version of rkt included in CoreOS is too old, it's fairly trivial to fetch the desired version [via a systemd unit][coreos-install-rkt].

## Debian

rkt is currently packaged in [Debian sid][pkg-debian] (unstable).

```
sudo apt-get install rkt
```

If you don't run sid, or wish for a newer version, you can [install manually](#deb-based).

## Fedora

Since Fedora version 24, rkt packages are available in the main repository. We recommend using recent Fedora releases or a manually installed package in order to have an up-to-date rkt binary.


```
sudo dnf install rkt
```

rkt's entry in the [Fedora package database][pkg-fedora] tracks packaging work for this distribution.

#### Caveat: SELinux

rkt does not work with the SELinux policies currently shipped with Fedora versions 24 and 25.

As a workaround, SELinux can be temporarily disabled:
```
sudo setenforce Permissive
```
Or permanently disabled by editing `/etc/selinux/config`:
```
SELINUX=permissive
```

#### Caveat: firewalld

Fedora uses [firewalld][firewalld] to dynamically define firewall zones.
rkt is [not yet fully integrated with firewalld][rkt-2206].
The default firewalld rules may interfere with the network connectivity of rkt pods.
To work around this, add a firewalld rule to allow pod traffic:
```
sudo firewall-cmd --add-source=172.16.28.0/24 --zone=trusted
```

172.16.28.0/24 is the subnet of the [default pod network][networking-overview-default]. The command must be adapted when rkt is configured to use a [different network][networking-overview-additional] with a different subnet.

## NixOS

rkt can be installed on NixOS using the following command:

```
nix-env -iA rkt
```

The source for the rkt.nix expression can be found on [GitHub][rkt-nixos]


## openSUSE

rkt is available in the [Virtualization:containers][rkt-opensuse] project on openSUSE Build Service.
Before installing, the appropriate repository needs to be added (usually Tumbleweed or Leap):

```
sudo zypper ar -f obs://Virtualization:containers/openSUSE_Tumbleweed/ virtualization_containers
sudo zypper ar -f obs://Virtualization:containers/openSUSE_Leap_42.1/ virtualization_containers
```

Install rkt using zypper:

```
sudo zypper in rkt
```

## Ubuntu

rkt is not packaged currently in Ubuntu. Instead, install manually using the 
[rkt debian package](#deb-based).

## Void

rkt is available in the [official binary packages][void-packages] for the Void Linux distribution.
The source for these packages is hosted on [GitHub][rkt-void].


# rkt-maintained packages
As part of the rkt build process, rpm and deb packages are built. If you need to use
the latest rkt version, or your distribution does not bundle rkt, these are available.

Currently the rkt upstream project does not maintain its own repository, so users of these packages must
upgrade manually.

### rpm-based 
```
gpg --recv-key 18AD5014C99EF7E3BA5F6CE950BDD3E0FC8A365E
wget https://github.com/coreos/rkt/releases/download/v1.25.0/rkt-1.25.0-1.x86_64.rpm
wget https://github.com/coreos/rkt/releases/download/v1.25.0/rkt-1.25.0-1.x86_64.rpm.asc
gpg --verify rkt-1.25.0-1.x86_64.rpm.asc
sudo rpm -Uvh rkt-1.25.0-1.x86_64.rpm
```

### deb-based
```
gpg --recv-key 18AD5014C99EF7E3BA5F6CE950BDD3E0FC8A365E
wget https://github.com/coreos/rkt/releases/download/v1.25.0/rkt_1.25.0-1_amd64.deb
wget https://github.com/coreos/rkt/releases/download/v1.25.0/rkt_1.25.0-1_amd64.deb.asc
gpg --verify rkt_1.25.0-1_amd64.deb.asc
sudo dpkg -i rkt_1.25.0-1_amd64.deb
```

[coreos-install-rkt]: install-rkt-in-coreos.md
[coreos-releases]: https://coreos.com/releases/
[debian-823322]: https://bugs.debian.org/cgi-bin/bugreport.cgi?bug=823322
[firewalld]: https://fedoraproject.org/wiki/FirewallD
[networking-overview-additional]: networking/overview.md#setting-up-additional-networks
[networking-overview-default]: networking/overview.md#the-default-network
[pkg-arch]: https://www.archlinux.org/packages/community/x86_64/rkt/
[pkg-centos]: https://cbs.centos.org/koji/packageinfo?packageID=4464
[pkg-debian]: https://packages.debian.org/sid/utils/rkt
[pkg-fedora]: https://admin.fedoraproject.org/pkgdb/package/rpms/rkt/
[rkt-nixos]: https://github.com/NixOS/nixpkgs/blob/master/pkgs/applications/virtualization/rkt/default.nix
[rkt-opensuse]: https://build.opensuse.org/package/show/Virtualization:containers/rkt
[rkt-void]: https://github.com/voidlinux/void-packages/tree/master/srcpkgs/rkt
[rkt-1305]: https://github.com/coreos/rkt/issues/1305
[rkt-1978]: https://github.com/coreos/rkt/issues/1978
[rkt-2206]: https://github.com/coreos/rkt/issues/2206
[rkt-2322]: https://github.com/coreos/rkt/issues/2322
[rkt-2325]: https://github.com/coreos/rkt/issues/2325
[rkt-2326]: https://github.com/coreos/rkt/issues/2326
[void-packages]: http://www.voidlinux.eu/packages/
