# iSCSI target container for testing.

Inspired by https://github.com/rvykydal/dockerfile-iscsid

* The container needs /lib/modules from the host to insert appropriate
  kernel modules for iscsi. This assumes that these modules are installed
  on the host!

* The container needs to run with docker --privileged

block.tar.gz is a small ext2 filesystem created by `create_block.sh` (run as root!)

