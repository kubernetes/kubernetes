## Multipath

To leverage multiple paths for block storage, it is important to perform the
multipath configuration on the host.
If your distribution does not provide `/etc/multipath.conf`, then you can
either use the following minimalistic one:

    defaults {
        find_multipaths yes
        user_friendly_names yes
    }

or create a new one by running:

    $ mpathconf --enable

Finally you'll need to ensure to start or reload and enable multipath:

    $ systemctl enable multipathd.service
    $ systemctl restart multipathd.service

**Note:** Any change to `multipath.conf` or enabling multipath can lead to
inaccessible block devices, because they'll be claimed by multipath and
exposed as a device in /dev/mapper/*.

Some additional informations about multipath can be found in the
[iSCSI documentation](iscsi/README.md)
