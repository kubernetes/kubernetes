# FAQ

## Initial Bootstrapping UX

etcd initial bootstrapping is done via command line flags such as
`--initial-cluster` or `--discovery`. These flags can safely be left on the
command line after your cluster is running but they will be ignored if you have
a non-empty data dir. So, why did we decide to have this sort of odd UX?

One of the design goals of etcd is easy bringup of clusters using a one-shot
static configuration like AWS Cloud Formation, PXE booting, etc. Essentially we
want to describe several virtual machines and bring them all up at once into an
etcd cluster.

To achieve this sort of hands-free cluster bootstrap we had two other options:

**API to bootstrap**

This is problematic because it cannot be coordinated from a single service file
and we didn't want to have the etcd socket listening but unresponsive to
clients for an unbound period of time.

It would look something like this:

```
ExecStart=/usr/bin/etcd
ExecStartPost/usr/bin/etcd init localhost:2379 --cluster=
```

**etcd init subcommand**

```
etcd init --cluster='default=http://localhost:2380,default=http://localhost:7001'...
etcd init --discovery https://discovery-example.etcd.io/193e4
```

Then after running an init step you would execute `etcd`. This however
introduced problems: we now have to define a hand-off protocol between the etcd
init process and the etcd binary itself. This is hard to coordinate in a single
service file such as:

```
ExecStartPre=/usr/bin/etcd init --cluster=....
ExecStart=/usr/bin/etcd
```

There are several error cases:

0) Init has already run and the data directory is already configured
1) Discovery fails because of network timeout, etc
2) Discovery fails because the cluster is already full and etcd needs to fall back to proxy
3) Static cluster configuration fails because of conflict, misconfiguration or timeout

In hindsight we could have made this work by doing:

```
rc	status
0	Init already ran
1	Discovery fails on network timeout, etc
0	Discovery fails for cluster full, coordinate via proxy state file
1	Static cluster configuration failed
```

Perhaps we can add the init command in a future version and deprecate if the UX
continues to confuse people.
