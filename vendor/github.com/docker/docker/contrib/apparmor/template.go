package main

const dockerProfileTemplate = `@{DOCKER_GRAPH_PATH}=/var/lib/docker

profile /usr/bin/docker (attach_disconnected, complain) {
  # Prevent following links to these files during container setup.
  deny /etc/** mkl,
  deny /dev/** kl,
  deny /sys/** mkl,
  deny /proc/** mkl,

  mount -> @{DOCKER_GRAPH_PATH}/**,
  mount -> /,
  mount -> /proc/**,
  mount -> /sys/**,
  mount -> /run/docker/netns/**,
  mount -> /.pivot_root[0-9]*/,

  / r,

  umount,
  pivot_root,
{{if ge .Version 209000}}
  signal (receive) peer=@{profile_name},
  signal (receive) peer=unconfined,
  signal (send),
{{end}}
  network,
  capability,
  owner /** rw,
  @{DOCKER_GRAPH_PATH}/** rwl,
  @{DOCKER_GRAPH_PATH}/linkgraph.db k,
  @{DOCKER_GRAPH_PATH}/network/files/boltdb.db k,
  @{DOCKER_GRAPH_PATH}/network/files/local-kv.db k,
  @{DOCKER_GRAPH_PATH}/[0-9]*.[0-9]*/linkgraph.db k,

  # For non-root client use:
  /dev/urandom r,
  /dev/null rw,
  /dev/pts/[0-9]* rw,
  /run/docker.sock rw,
  /proc/** r,
  /proc/[0-9]*/attr/exec w,
  /sys/kernel/mm/hugepages/ r,
  /etc/localtime r,
  /etc/ld.so.cache r,
  /etc/passwd r,

{{if ge .Version 209000}}
  ptrace peer=@{profile_name},
  ptrace (read) peer=docker-default,
  deny ptrace (trace) peer=docker-default,
  deny ptrace peer=/usr/bin/docker///bin/ps,
{{end}}

  /usr/lib/** rm,
  /lib/** rm,

  /usr/bin/docker pix,
  /sbin/xtables-multi rCx,
  /sbin/iptables rCx,
  /sbin/modprobe rCx,
  /sbin/auplink rCx,
  /sbin/mke2fs rCx,
  /sbin/tune2fs rCx,
  /sbin/blkid rCx,
  /bin/kmod rCx,
  /usr/bin/xz rCx,
  /bin/ps rCx,
  /bin/tar rCx,
  /bin/cat rCx,
  /sbin/zfs rCx,
  /sbin/apparmor_parser rCx,

{{if ge .Version 209000}}
  # Transitions
  change_profile -> docker-*,
  change_profile -> unconfined,
{{end}}

  profile /bin/cat (complain) {
    /etc/ld.so.cache r,
    /lib/** rm,
    /dev/null rw,
    /proc r,
    /bin/cat mr,

    # For reading in 'docker stats':
    /proc/[0-9]*/net/dev r,
  }
  profile /bin/ps (complain) {
    /etc/ld.so.cache r,
    /etc/localtime r,
    /etc/passwd r,
    /etc/nsswitch.conf r,
    /lib/** rm,
    /proc/[0-9]*/** r,
    /dev/null rw,
    /bin/ps mr,

{{if ge .Version 209000}}
    # We don't need ptrace so we'll deny and ignore the error.
    deny ptrace (read, trace),
{{end}}

    # Quiet dac_override denials
    deny capability dac_override,
    deny capability dac_read_search,
    deny capability sys_ptrace,

    /dev/tty r,
    /proc/stat r,
    /proc/cpuinfo r,
    /proc/meminfo r,
    /proc/uptime r,
    /sys/devices/system/cpu/online r,
    /proc/sys/kernel/pid_max r,
    /proc/ r,
    /proc/tty/drivers r,
  }
  profile /sbin/iptables (complain) {
{{if ge .Version 209000}}
    signal (receive) peer=/usr/bin/docker,
{{end}}
    capability net_admin,
  }
  profile /sbin/auplink flags=(attach_disconnected, complain) {
{{if ge .Version 209000}}
    signal (receive) peer=/usr/bin/docker,
{{end}}
    capability sys_admin,
    capability dac_override,

    @{DOCKER_GRAPH_PATH}/aufs/** rw,
    @{DOCKER_GRAPH_PATH}/tmp/** rw,
    # For user namespaces:
    @{DOCKER_GRAPH_PATH}/[0-9]*.[0-9]*/** rw,

    /sys/fs/aufs/** r,
    /lib/** rm,
    /apparmor/.null r,
    /dev/null rw,
    /etc/ld.so.cache r,
    /sbin/auplink rm,
    /proc/fs/aufs/** rw,
    /proc/[0-9]*/mounts rw,
  }
  profile /sbin/modprobe /bin/kmod (complain) {
{{if ge .Version 209000}}
    signal (receive) peer=/usr/bin/docker,
{{end}}
    capability sys_module,
    /etc/ld.so.cache r,
    /lib/** rm,
    /dev/null rw,
    /apparmor/.null rw,
    /sbin/modprobe rm,
    /bin/kmod rm,
    /proc/cmdline r,
    /sys/module/** r,
    /etc/modprobe.d{/,/**} r,
  }
  # xz works via pipes, so we do not need access to the filesystem.
  profile /usr/bin/xz (complain) {
{{if ge .Version 209000}}
    signal (receive) peer=/usr/bin/docker,
{{end}}
    /etc/ld.so.cache r,
    /lib/** rm,
    /usr/bin/xz rm,
    deny /proc/** rw,
    deny /sys/** rw,
  }
  profile /sbin/xtables-multi (attach_disconnected, complain) {
    /etc/ld.so.cache r,
    /lib/** rm,
    /sbin/xtables-multi rm,
    /apparmor/.null w,
    /dev/null rw,

    /proc r,

    capability net_raw,
    capability net_admin,
    network raw,
  }
  profile /sbin/zfs (attach_disconnected, complain) {
    file,
    capability,
  }
  profile /sbin/mke2fs (complain) {
    /sbin/mke2fs rm,

    /lib/** rm,

    /apparmor/.null w,

    /etc/ld.so.cache r,
    /etc/mke2fs.conf r,
    /etc/mtab r,

    /dev/dm-* rw,
    /dev/urandom r,
    /dev/null rw,

    /proc/swaps r,
    /proc/[0-9]*/mounts r,
  }
  profile /sbin/tune2fs (complain) {
    /sbin/tune2fs rm,

    /lib/** rm,

    /apparmor/.null w,

    /etc/blkid.conf r,
    /etc/mtab r,
    /etc/ld.so.cache r,

    /dev/null rw,
    /dev/.blkid.tab r,
    /dev/dm-* rw,

    /proc/swaps r,
    /proc/[0-9]*/mounts r,
  }
  profile /sbin/blkid (complain) {
    /sbin/blkid rm,

    /lib/** rm,
    /apparmor/.null w,

    /etc/ld.so.cache r,
    /etc/blkid.conf r,

    /dev/null rw,
    /dev/.blkid.tab rl,
    /dev/.blkid.tab* rwl,
    /dev/dm-* r,

    /sys/devices/virtual/block/** r,

    capability mknod,

    mount -> @{DOCKER_GRAPH_PATH}/**,
  }
  profile /sbin/apparmor_parser (complain) {
    /sbin/apparmor_parser rm,

    /lib/** rm,

    /etc/ld.so.cache r,
    /etc/apparmor/** r,
    /etc/apparmor.d/** r,
    /etc/apparmor.d/cache/** w,

    /dev/null rw,

    /sys/kernel/security/apparmor/** r,
    /sys/kernel/security/apparmor/.replace w,

    /proc/[0-9]*/mounts r,
    /proc/sys/kernel/osrelease r,
    /proc r,

    capability mac_admin,
  }
}`
