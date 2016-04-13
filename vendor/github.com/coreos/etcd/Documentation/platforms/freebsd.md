# FreeBSD

Starting with version 0.1.2 both etcd and etcdctl have been ported to FreeBSD and can
be installed either via packages or ports system. Their versions have been recently
updated to 0.2.0 so now you can enjoy using etcd and etcdctl on FreeBSD 10.0 (RC4 as
of now) and 9.x where they have been tested. They might also work when installed from
ports on earlier versions of FreeBSD, but your mileage may vary.

## Installation

### Using pkgng package system

1. If you do not have pkg­ng installed, install it with command `pkg` and answering 'Y'
when asked

2. Update your repository data with `pkg update`

3. Install etcd with `pkg install coreos-etcd coreos-etcdctl`

4. Verify successful installation with `pkg info | grep etcd` and you should get:

```
r@fbsd­10:/ # pkg info | grep etcd
coreos­etcd­0.2.0              Highly­available key value store and service discovery
coreos­etcdctl­0.2.0           Simple commandline client for etcd
r@fbsd­10:/ #
```

5. You’re ready to use etcd and etcdctl! For more information about using pkgng, please
see: http://www.freebsd.org/doc/handbook/pkgng­intro.html
 
### Using ports system

1. If you do not have ports installed, install with with `portsnap fetch extract` (it
may take some time depending on your hardware and network connection)

2. Build etcd with `cd /usr/ports/devel/etcd && make install clean`, you
will get an option to build and install documentation and etcdctl with it.

3. If you haven't installed it with etcdctl, and you would like to install it later, you can build it
with `cd /usr/ports/devel/etcdctl && make install clean`

4. Verify successful installation with `pkg info | grep etcd` and you should get:
 

```
r@fbsd­10:/ # pkg info | grep etcd
coreos­etcd­0.2.0              Highly­available key value store and service discovery
coreos­etcdctl­0.2.0           Simple commandline client for etcd
r@fbsd­10:/ #
```

5. You’re ready to use etcd and etcdctl! For more information about using ports system,
please see: https://www.freebsd.org/doc/handbook/ports­using.html

## Issues

If you find any issues with the build/install procedure or you've found a problem that
you've verified is local to FreeBSD version only (for example, by not being able to
reproduce it on any other platform, like OSX or Linux), please sent a
problem report using this page for more
information: http://www.freebsd.org/send­pr.html
