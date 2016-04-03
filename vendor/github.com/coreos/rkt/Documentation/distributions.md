# Installing rkt on popular Linux distributions

## CoreOS

rkt is an integral part of CoreOS, installed with the operating system.
The [CoreOS releases page](https://coreos.com/releases/) lists the version of rkt available in each CoreOS release channel.

## Fedora

rkt is packaged in the development version of Fedora, [Rawhide](https://fedoraproject.org/wiki/Releases/Rawhide):
```
sudo dnf install rkt
```

Until the rkt package makes its way into the general Fedora releases, [download the latest rkt directly from the project](https://github.com/coreos/rkt/releases).

rkt's entry in the [Fedora package database](https://admin.fedoraproject.org/pkgdb/package/rpms/rkt/) tracks packaging work for this distribution.

#### Caveat: SELinux

rkt currently does not integrate with SELinux on Fedora.
See [#1727](https://github.com/coreos/rkt/issues/1727).
As a workaround, SELinux can be temporarily disabled:
```
sudo setenforce Permissive
```
Or permanently disabled by editing `/etc/selinux/config`:
```
SELINUX=permissive
```

#### Caveat: firewall

The default firewall rules can block the traffic from rkt pods.
As a workaround, they can be removed:
```
sudo iptables -F
sudo iptables -F -t nat
```

## Arch

rkt is available in the [Arch User Repository (AUR)](https://aur.archlinux.org/packages/rkt).
Installing instructions are available in the [AUR installing packages documentation](https://wiki.archlinux.org/index.php/Arch_User_Repository#Installing_packages) or you can use an [AUR helper](https://wiki.archlinux.org/index.php/AUR_helpers).

## Void

rkt is available in the [official binary packages](http://www.voidlinux.eu/packages/) for the Void Linux distribution.
The source for these packages is hosted on [GitHub](https://github.com/voidlinux/void-packages/tree/master/srcpkgs/rkt).
