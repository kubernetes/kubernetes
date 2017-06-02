## Vitess Example

This example shows how to run a [Vitess](http://vitess.io) cluster in Kubernetes.
Vitess is a MySQL clustering system developed at YouTube that makes sharding
transparent to the application layer. It also makes scaling MySQL within
Kubernetes as simple as launching more pods.

The example brings up a database with 2 shards, and then runs a pool of
[sharded guestbook](https://github.com/youtube/vitess/tree/master/examples/kubernetes/guestbook)
pods. The guestbook app was ported from the original
[guestbook](../../../examples/guestbook-go/)
example found elsewhere in this tree, modified to use Vitess as the backend.

For a more detailed, step-by-step explanation of this example setup, see the
[Vitess on Kubernetes](http://vitess.io/getting-started/) guide.

### Prerequisites

You'll need to install [Go 1.4+](https://golang.org/doc/install) to build
`vtctlclient`, the command-line admin tool for Vitess.

We also assume you have a running Kubernetes cluster with `kubectl` pointing to
it by default. See the [Getting Started guides](https://kubernetes.io/docs/getting-started-guides/)
for how to get to that point. Note that your Kubernetes cluster needs to have
enough resources (CPU+RAM) to schedule all the pods. By default, this example
requires a cluster-wide total of at least 6 virtual CPUs and 10GiB RAM. You can
tune these requirements in the
[resource limits](https://kubernetes.io/docs/user-guide/compute-resources.md)
section of each YAML file.

Lastly, you need to open ports 30000-30001 (for the Vitess admin daemon) and 80 (for
the guestbook app) in your firewall. See the
[Services and Firewalls](https://kubernetes.io/docs/user-guide/services-firewalls.md)
guide for examples of how to do that.

### Configure site-local settings

Run the `configure.sh` script to generate a `config.sh` file, which will be used
to customize your cluster settings.

``` console
./configure.sh
```

Currently, we have out-of-the-box support for storing
[backups](http://vitess.io/user-guide/backup-and-restore.html) in
[Google Cloud Storage](https://cloud.google.com/storage/).
If you're using GCS, fill in the fields requested by the configure script.
Note that your Kubernetes cluster must be running on instances with the
`storage-rw` scope for this to work. With Container Engine, you can do this by
passing `--scopes storage-rw` to the `glcoud container clusters create` command.

For other platforms, you'll need to choose the `file` backup storage plugin,
and mount a read-write network volume into the `vttablet` and `vtctld` pods.
For example, you can mount any storage service accessible through NFS into a
Kubernetes volume. Then provide the mount path to the configure script here.

If you prefer to skip setting up a backup volume for the purpose of this example,
you can choose `file` mode and set the path to `/tmp`.

### Start Vitess

``` console
./vitess-up.sh
```

This will run through the steps to bring up Vitess. At the end, you should see
something like this:

``` console
****************************
* Complete!
* Use the following line to make an alias to kvtctl:
* alias kvtctl='$GOPATH/bin/vtctlclient -server 104.197.47.173:30001'
* See the vtctld UI at: http://104.197.47.173:30000
****************************
```

### Start the Guestbook app

``` console
./guestbook-up.sh
```

The guestbook service is configured with `type: LoadBalancer` to tell Kubernetes
to expose it on an external IP. It may take a minute to set up, but you should
soon see the external IP show up under the internal one like this:

``` console
$ kubectl get service guestbook
NAME        LABELS    SELECTOR         IP(S)             PORT(S)
guestbook   <none>    name=guestbook   10.67.253.173     80/TCP
                                       104.197.151.132
```

Visit the external IP in your browser to view the guestbook. Note that in this
modified guestbook, there are multiple pages to demonstrate range-based sharding
in Vitess. Each page number is assigned to one of the shards using a
[consistent hashing](https://en.wikipedia.org/wiki/Consistent_hashing) scheme.

### Tear down

``` console
./guestbook-down.sh
./vitess-down.sh
```

You may also want to remove any firewall rules you created.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/storage/vitess/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
