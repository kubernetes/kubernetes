# ipc-utils

This container will be used to exercise the HostIPC functionality in
e2e-node tests.

The version of `ipcs` shipped in busybox performs operations that get
blocked by SELinux on hosts where it is enabled. The version of `ipcs`
in util-linux does not perform those operations, rather it checks
whether the /proc files it needs are available and proceeds to read
from them directly.

Using `ipcs` from util-linux makes these tests pass, even when running
under SELinux enabled, so let's use them here.