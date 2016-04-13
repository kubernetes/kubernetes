<!--[metadata]>
+++
draft = true
title = "File system"
description = "How Linux organizes its persistent storage"
keywords = ["containers, files,  linux"]
[menu.main]
parent = "mn_reference"
+++
<![end-metadata]-->

# File system

## Introduction

![](/terms/images/docker-filesystems-generic.png)

In order for a Linux system to run, it typically needs two [file
systems](http://en.wikipedia.org/wiki/Filesystem):

1. boot file system (bootfs)
2. root file system (rootfs)

The **boot file system** contains the bootloader and the kernel. The
user never makes any changes to the boot file system. In fact, soon
after the boot process is complete, the entire kernel is in memory, and
the boot file system is unmounted to free up the RAM associated with the
initrd disk image.

The **root file system** includes the typical directory structure we
associate with Unix-like operating systems:
`/dev, /proc, /bin, /etc, /lib, /usr,` and `/tmp` plus all the configuration
files, binaries and libraries required to run user applications (like bash,
ls, and so forth).

While there can be important kernel differences between different Linux
distributions, the contents and organization of the root file system are
usually what make your software packages dependent on one distribution
versus another. Docker can help solve this problem by running multiple
distributions at the same time.

![](/terms/images/docker-filesystems-multiroot.png)
