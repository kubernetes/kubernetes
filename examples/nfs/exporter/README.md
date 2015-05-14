# NFS-exporter container

Inspired by https://github.com/cpuguy83/docker-nfs-server. Rewritten for
Fedora.

Serves NFS4 exports, defined on command line. At least one export must be defined!

Usage::

    docker run -d --name nfs --privileged jsafrane/nfsexporter /path/to/share /path/to/share2 ...


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/nfs/exporter/README.md?pixel)]()
