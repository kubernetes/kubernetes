<!--[metadata]>
+++
title = "Installation on Rackspace Cloud"
description = "Installation instructions for Docker on Rackspace Cloud."
keywords = ["Rackspace Cloud, installation, docker, linux,  ubuntu"]
[menu.main]
parent = "smn_cloud"
+++
<![end-metadata]-->

# Rackspace Cloud

Installing Docker on Ubuntu provided by Rackspace is pretty
straightforward, and you should mostly be able to follow the
[*Ubuntu*](../ubuntulinux/#ubuntu-linux) installation guide.

**However, there is one caveat:**

If you are using any Linux not already shipping with the 3.8 kernel you
will need to install it. And this is a little more difficult on
Rackspace.

Rackspace boots their servers using grub's `menu.lst`
and does not like non `virtual` packages (e.g., Xen compatible)
kernels there, although they do work. This results in
`update-grub` not having the expected result, and
you will need to set the kernel manually.

**Do not attempt this on a production machine!**

    # update apt
    $ apt-get update

    # install the new kernel
    $ apt-get install linux-generic-lts-raring

Great, now you have the kernel installed in `/boot/`, next you need to
make it boot next time.

    # find the exact names
    $ find /boot/ -name '*3.8*'

    # this should return some results

Now you need to manually edit `/boot/grub/menu.lst`,
you will find a section at the bottom with the existing options. Copy
the top one and substitute the new kernel into that. Make sure the new
kernel is on top, and double check the kernel and initrd lines point to
the right files.

Take special care to double check the kernel and initrd entries.

    # now edit /boot/grub/menu.lst
    $ vi /boot/grub/menu.lst

It will probably look something like this:

    ## ## End Default Options ##

    title              Ubuntu 12.04.2 LTS, kernel 3.8.x generic
    root               (hd0)
    kernel             /boot/vmlinuz-3.8.0-19-generic root=/dev/xvda1 ro quiet splash console=hvc0
    initrd             /boot/initrd.img-3.8.0-19-generic

    title              Ubuntu 12.04.2 LTS, kernel 3.2.0-38-virtual
    root               (hd0)
    kernel             /boot/vmlinuz-3.2.0-38-virtual root=/dev/xvda1 ro quiet splash console=hvc0
    initrd             /boot/initrd.img-3.2.0-38-virtual

    title              Ubuntu 12.04.2 LTS, kernel 3.2.0-38-virtual (recovery mode)
    root               (hd0)
    kernel             /boot/vmlinuz-3.2.0-38-virtual root=/dev/xvda1 ro quiet splash  single
    initrd             /boot/initrd.img-3.2.0-38-virtual

Reboot the server (either via command line or console)

    # reboot

Verify the kernel was updated

    $ uname -a
    # Linux docker-12-04 3.8.0-19-generic #30~precise1-Ubuntu SMP Wed May 1 22:26:36 UTC 2013 x86_64 x86_64 x86_64 GNU/Linux

    # nice! 3.8.

Now you can finish with the [*Ubuntu*](../ubuntulinux/#ubuntu-linux)
instructions.
