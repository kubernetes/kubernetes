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


####A Records and hostname based on Pod's hostname and subdomain fields
Currently when a pod is created, its hostname is the Pod's `metadata.name` value.

With v1.2, users can specify a Pod annotation, `pod.beta.kubernetes.io/hostname`, to specify what the Pod's hostname should be.
If the annotation is specified, the annotation value takes precendence over the Pod's name, to be the hostname of the pod.
For example, given a Pod with annotation `pod.beta.kubernetes.io/hostname: my-pod-name`, the Pod will have its hostname set to "my-pod-name".

With v1.3, the PodSpec has a `hostname` field, which can be used to specify the Pod's hostname. This field value takes precedence over the
`pod.beta.kubernetes.io/hostname` annotation value.

v1.2 introduces a beta feature where the user can specify a Pod annotation, `pod.beta.kubernetes.io/subdomain`, to specify what the Pod's subdomain should be.
If the annotation is specified, the fully qualified Pod hostname will be "<hostname>.<subdomain>.<pod namespace>.svc.<cluster domain>".
For example, given a Pod with the hostname annotation set to "foo", and the subdomain annotation set to "bar", in namespace "my-namespace", the pod will set its own FQDN as "foo.bar.my-namespace.svc.cluster.local"

With v1.3, the PodSpec has a `subdomain` field, which can be used to specify the Pod's subdomain. This field value takes precedence over the
`pod.beta.kubernetes.io/subdomain` annotation value.

Example:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: busybox
  namespace: default
spec:
  hostname: busybox-1
  subdomain: default
  containers:
  - image: busybox
    command:
      - sleep
      - "3600"
    name: busybox
```

If there exists a headless service in the same namespace as the pod and with the same name as the subdomain, the cluster's KubeDNS Server will also return an A record for the Pod's fully qualified hostname.
Given a Pod with the hostname set to "foo" and the subdomain set to "bar", and a headless Service named "bar" in the same namespace, the pod will see it's own FQDN as "foo.bar.my-namespace.svc.cluster.local". DNS will serve an A record at that name, pointing to the Pod's IP.

With v1.2, the Endpoints object also has a new annotation `endpoints.beta.kubernetes.io/hostnames-map`. Its value is the json representation of map[string(IP)][endpoints.HostRecord], for example: '{"10.245.1.6":{HostName: "my-webserver"}}'.
If the Endpoints are for a headless service, then A records will be created with the format <hostname>.<service name>.<pod namespace>.svc.<cluster domain>
For the example json, if endpoints are for a headless service named "bar", and one of the endpoints has IP "10.245.1.6", then a A record will be created with the name "my-webserver.bar.my-namespace.svc.cluster.local" and the A record lookup would return "10.245.1.6".
This endpoints annotation generally does not need to be specified by end-users, but can used by the internal service controller to deliver the aforementioned feature.

With v1.3, The Endpoints object can specify the `hostname` for any endpoint, along with its IP. The hostname field takes precedence over the hostname value
that might have been specified via the  `endpoints.beta.kubernetes.io/hostnames-map` annotation.

With v1.3, the following annotations are deprecated: `pod.beta.kubernetes.io/hostname`, `pod.beta.kubernetes.io/subdomain`, `endpoints.beta.kubernetes.io/hostnames-map`

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
the example files ([ReplicationController](../../cluster/addns/dns/skydns-rc.yaml.in) and
[Service](../../cluster/addons/dns/skydns-svc.yaml.in)), but keep in mind that these are templated for
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
NAME      READY     STATUS    RESTARTS   AGE
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
<del>SkyDNS depends on etcd for what to serve, but it doesn't really need all of
what etcd offers (at least not in the way we use it).  For simplicity, we run
etcd and SkyDNS together in a pod, and we do not try to link etcd instances
across replicas.  A helper container called [kube2sky](kube2sky/) also runs in
the pod and acts a bridge between Kubernetes and SkyDNS.  It finds the
Kubernetes master through the `kubernetes` service (via environment
variables), pulls service info from the master, and writes that to etcd for
SkyDNS to find.</del>

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
The container containing the kube-dns binary needs to be built for every
architecture and pushed to the registry manually whenever the kube-dns binary
has code changes. Every significant change to the functionality should result
in a bump of the TAG in the Makefile.

Any significant changes to the YAML template for `kube-dns` should result a bump
of the version number for the `kube-dns` replication controller and well as the
`version` label. This will permit a rolling update of `kube-dns`.


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/build/kube-dns/README.md?pixel)]()
