# rkt and SELinux

rkt supports running containers using SELinux [SVirt][svirt].
At start-up, rkt will attempt to read `/etc/selinux/(policy)/contexts/lxc_contexts`.
If this file doesn't exist, no SELinux transitions will be performed.
If it does, rkt will generate a per-instance context.
All mounts for the instance will be created using the file context defined in `lxc_contexts`, and the instance processes will be run in a context derived from the process context defined in `lxc_contexts`.

Processes started in these contexts will be unable to interact with processes or files in any other instance's context, even though they are running as the same user.
Individual Linux distributions may impose additional isolation constraints on these contexts - please refer to your distribution documentation for further details.


[svirt]: http://selinuxproject.org/page/SVirt
