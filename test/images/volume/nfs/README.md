# NFS server container for testing

This container exports '/' directory with an index.html inside. NFSv4 only.

Accepts a -G option for specifying a group id to give exported directories.
Clients in the specified group will have full rwx permissions, others none.

Inspired by https://github.com/cpuguy83/docker-nfs-server.

Used by test/e2e/* to test NFSVolumeSource. Not for production use!

