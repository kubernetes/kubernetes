# DNS in Kubernetes

Kubernetes offers a DNS cluster addon, which most of the supported environments
enable by default.  We use [SkyDNS](https://github.com/skynetservices/skydns)
as the DNS server, with some custom logic to slave it to the kubernetes API
server.

## What things get DNS names?
The only objects to which we are assigning DNS names are Services.  Every
Kubernetes Service is assigned a virtual IP address which is stable as long as
the Service exists (as compared to Pod IPs which can change over time due to
crashes or scheduling changes).  This maps well to DNS, which has a long
history of clients that, on purpose or on accident, do not respect DNS TTLs
(see previous remark about Pod IPs changing).

## Where does resolution work?
Kubernetes Service DNS names can be resolved using standard methods (e.g. [`gethostbyname`](
http://linux.die.net/man/3/gethostbyname)) inside any pod, except pods which
have the `hostNetwork` field set to `true`.

## Supported DNS schema
The following sections detail the supported record types and layout that is
supported.  Any other layout or names or queries that happen to work are
considered implementation details and are subject to change without warning.

### Services

#### A records
"Normal" (not headless) Services are assigned a DNS A record for a name of the
form `my-svc.my-namespace.svc.cluster.local`.  This resolves to the cluster IP
of the Service.

"Headless" (without a cluster IP) Services are also assigned a DNS A record for
a name of the form `my-svc.my-namespace.svc.cluster.local`.  Unlike normal
Services, this resolves to the set of IPs of the pods selected by the Service.
Clients are expected to consume the set or else use standard round-robin
selection from the set.

### SRV records
SRV Records are created for named ports that are part of normal or Headless
Services.
For each named port, the SRV record would have the form
`_my-port-name._my-port-protocol.my-svc.my-namespace.svc.cluster.local`.
For a regular service, this resolves to the port number and the CNAME:
`my-svc.my-namespace.svc.cluster.local`.
For a headless service, this resolves to multiple answers, one for each pod
that is backing the service, and contains the port number and a CNAME of the pod
of the form `auto-generated-name.my-svc.my-namespace.svc.cluster.local`.

### Backwards compatibility
Previous versions of kube-dns made names of the for
`my-svc.my-namespace.cluster.local` (the 'svc' level was added later).  This
is no longer supported.

### Pods

#### A Records
When enabled, pods are assigned a DNS A record in the form of `pod-ip-address.my-namespace.pod.cluster.local`.

For example, a pod with ip `1.2.3.4` in the namespace `default` with a dns name of `cluster.local` would have an entry: `1-2-3-4.default.pod.cluster.local`.

## How do I find the DNS server?
The DNS server itself runs as a Kubernetes Service.  This gives it a stable IP
address.  When you run the SkyDNS service, you want to assign a static IP to use for
the Service.  For example, if you assign the DNS Service IP as `10.0.0.10`, you
can configure your kubelet to pass that on to each container as a DNS server.

Of course, giving services a name is just half of the problem - DNS names need a
domain also.  This implementation uses a configurable local domain, which can
also be passed to containers by kubelet as a DNS search suffix.

## How do I configure it?
The easiest way to use DNS is to use a supported kubernetes cluster setup,
which should have the required logic to read some config variables and plumb
them all the way down to kubelet.

Supported environments offer the following config flags, which are used at
cluster turn-up to create the SkyDNS pods and configure the kubelets.  For
example, see `cluster/gce/config-default.sh`.

```sh
ENABLE_CLUSTER_DNS="${KUBE_ENABLE_CLUSTER_DNS:-true}"
DNS_SERVER_IP="10.0.0.10"
DNS_DOMAIN="cluster.local"
DNS_REPLICAS=1
```

This enables DNS with a DNS Service IP of `10.0.0.10` and a local domain of
`cluster.local`, served by a single copy of SkyDNS.

If you are not using a supported cluster setup, you will have to replicate some
of this yourself.  First, each kubelet needs to run with the following flags
set:

```
--cluster-dns=<DNS service ip>
--cluster-domain=<default local domain>
```

Second, you need to start the DNS server ReplicationController and Service. See
the example files ([ReplicationController](skydns-rc.yaml.in) and
[Service](skydns-svc.yaml.in)), but keep in mind that these are templated for
Salt.  You will need to replace the `{{ <param> }}` blocks with your own values
for the config variables mentioned above.  Other than the templating, these are
normal kubernetes objects, and can be instantiated with `kubectl create`.

## How do I test if it is working?
First deploy DNS as described above.

### 1 Create a simple Pod to use as a test environment.

Create a file named busybox.yaml with the
following contents:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: busybox
  namespace: default
spec:
  containers:
  - image: busybox
    command:
      - sleep
      - "3600"
    imagePullPolicy: IfNotPresent
    name: busybox
  restartPolicy: Always
```

Then create a pod using this file:

```
kubectl create -f busybox.yaml
```

### 2 Wait for this pod to go into the running state.

You can get its status with:
```
kubectl get pods busybox
```

You should see:
```
NAME      READY     REASON    RESTARTS   AGE
busybox   1/1       Running   0          <some-time>
```

### 3 Validate DNS works
Once that pod is running, you can exec nslookup in that environment:
```
kubectl exec busybox -- nslookup kubernetes.default
```

You should see something like:
```
Server:    10.0.0.10
Address 1: 10.0.0.10

Name:      kubernetes.default
Address 1: 10.0.0.1
```

If you see that, DNS is working correctly.


## How does it work?
SkyDNS depends on etcd for what to serve, but it doesn't really need all of
what etcd offers (at least not in the way we use it).  For simplicity, we run
etcd and SkyDNS together in a pod, and we do not try to link etcd instances
across replicas.  A helper container called [kube2sky](kube2sky/) also runs in
the pod and acts a bridge between Kubernetes and SkyDNS.  It finds the
Kubernetes master through the `kubernetes` service (via environment
variables), pulls service info from the master, and writes that to etcd for
SkyDNS to find.

## Inheriting DNS from the node
When running a pod, kubelet will prepend the cluster DNS server and search
paths to the node's own DNS settings.  If the node is able to resolve DNS names
specific to the larger environment, pods should be able to, also.  See "Known
issues" below for a caveat.

If you don't want this, or if you want a different DNS config for pods, you can
use the kubelet's `--resolv-conf` flag.  Setting it to "" means that pods will
not inherit DNS.  Setting it to a valid file path means that kubelet will use
this file instead of `/etc/resolv.conf` for DNS inheritance.

## Known issues
Kubernetes installs do not configure the nodes' resolv.conf files to use the
cluster DNS by default, because that process is inherently distro-specific.
This should probably be implemented eventually.

Linux's libc is impossibly stuck ([see this bug from
2005](https://bugzilla.redhat.com/show_bug.cgi?id=168253)) with limits of just
3 DNS `nameserver` records and 6 DNS `search` records.  Kubernetes needs to
consume 1 `nameserver` record and 3 `search` records.  This means that if a
local installation already uses 3 `nameserver`s or uses more than 3 `search`es,
some of those settings will be lost.  As a partial workaround, the node can run
`dnsmasq` which will provide more `nameserver` entries, but not more `search`
entries.  You can also use kubelet's `--resolv-conf` flag.

## Making changes
Please observe the release process for making changes to the `kube2sky`
image that is documented in [RELEASES.md](kube2sky/RELEASES.md). Any significant changes
to the YAML template for `kube-dns` should result a bump of the version number
for the `kube-dns` replication controller and well as the `version` label. This
will permit a rolling update of `kube-dns`.


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/cluster/addons/dns/README.md?pixel)]()
