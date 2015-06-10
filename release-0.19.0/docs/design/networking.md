# Networking

## Model and motivation

Kubernetes deviates from the default Docker networking model.  The goal is for each pod to have an IP in a flat shared networking namespace that has full communication with other physical computers and containers across the network.  IP-per-pod creates a clean, backward-compatible model where pods can be treated much like VMs or physical hosts from the perspectives of port allocation, networking, naming, service discovery, load balancing, application configuration, and migration.

OTOH, dynamic port allocation requires supporting both static ports (e.g., for externally accessible services) and dynamically allocated ports, requires partitioning centrally allocated and locally acquired dynamic ports, complicates scheduling (since ports are a scarce resource), is inconvenient for users, complicates application configuration, is plagued by port conflicts and reuse and exhaustion, requires non-standard approaches to naming (e.g., etcd rather than DNS), requires proxies and/or redirection for programs using standard naming/addressing mechanisms (e.g., web browsers), requires watching and cache invalidation for address/port changes for instances in addition to watching group membership changes, and obstructs container/pod migration (e.g., using CRIU). NAT introduces additional complexity by fragmenting the addressing space, which breaks self-registration mechanisms, among other problems.

With the IP-per-pod model, all user containers within a pod behave as if they are on the same host with regard to networking. They can all reach each other’s ports on localhost. Ports which are published to the host interface are done so in the normal Docker way. All containers in all pods can talk to all other containers in all other pods by their 10-dot addresses.

In addition to avoiding the aforementioned problems with dynamic port allocation, this approach reduces friction for applications moving from the world of uncontainerized apps on physical or virtual hosts to containers within pods. People running application stacks together on the same host have already figured out how to make ports not conflict (e.g., by configuring them through environment variables) and have arranged for clients to find them.

The approach does reduce isolation between containers within a pod -- ports could conflict, and there couldn't be private ports across containers within a pod, but applications requiring their own port spaces could just run as separate pods and processes requiring private communication could run within the same container. Besides, the premise of pods is that containers within a pod share some resources (volumes, cpu, ram, etc.) and therefore expect and tolerate reduced isolation. Additionally, the user can control what containers belong to the same pod whereas, in general, they don't control what pods land together on a host.

When any container calls SIOCGIFADDR, it sees the IP that any peer container would see them coming from -- each pod has its own IP address that other pods can know. By making IP addresses and ports the same within and outside the containers and pods, we create a NAT-less, flat address space. "ip addr show" should work as expected. This would enable all existing naming/discovery mechanisms to work out of the box, including self-registration mechanisms and applications that distribute IP addresses. (We should test that with etcd and perhaps one other option, such as Eureka (used by Acme Air) or Consul.) We should be optimizing for inter-pod network communication. Within a pod, containers are more likely to use communication through volumes (e.g., tmpfs) or IPC.

This is different from the standard Docker model. In that mode, each container gets an IP in the 172-dot space and would only see that 172-dot address from SIOCGIFADDR. If these containers connect to another container the peer would see the connect coming from a different IP than the container itself knows. In short - you can never self-register anything from a container, because a container can not be reached on its private IP.

An alternative we considered was an additional layer of addressing: pod-centric IP per container. Each container would have its own local IP address, visible only within that pod. This would perhaps make it easier for containerized applications to move from physical/virtual hosts to pods, but would be more complex to implement (e.g., requiring a bridge per pod, split-horizon/VP DNS) and to reason about, due to the additional layer of address translation, and would break self-registration and IP distribution mechanisms.

## Current implementation

For the Google Compute Engine cluster configuration scripts, [advanced routing](https://developers.google.com/compute/docs/networking#routing) is set up so that each VM has an extra 256 IP addresses that get routed to it.  This is in addition to the 'main' IP address assigned to the VM that is NAT-ed for Internet access.  The networking bridge (called `cbr0` to differentiate it from `docker0`) is set up outside of Docker proper and only does NAT for egress network traffic that isn't aimed at the virtual network.

Ports mapped in from the 'main IP' (and hence the internet if the right firewall rules are set up) are proxied in user mode by Docker.  In the future, this should be done with `iptables` by either the Kubelet or Docker: [Issue #15](https://github.com/GoogleCloudPlatform/kubernetes/issues/15).

We start Docker with:
    DOCKER_OPTS="--bridge cbr0 --iptables=false"

We set up this bridge on each node with SaltStack, in [container_bridge.py](cluster/saltbase/salt/_states/container_bridge.py).

    cbr0:
      container_bridge.ensure:
      - cidr: {{ grains['cbr-cidr'] }}
    ...
    grains:
      roles:
      - kubernetes-pool
      cbr-cidr: $MINION_IP_RANGE

We make these addresses routable in GCE:

    gcloud compute routes add "${MINION_NAMES[$i]}" \
      --project "${PROJECT}" \
      --destination-range "${MINION_IP_RANGES[$i]}" \
      --network "${NETWORK}" \
      --next-hop-instance "${MINION_NAMES[$i]}" \
      --next-hop-instance-zone "${ZONE}" &

The minion IP ranges are /24s in the 10-dot space.

GCE itself does not know anything about these IPs, though.

These are not externally routable, though, so containers that need to communicate with the outside world need to use host networking. To set up an external IP that forwards to the VM, it will only forward to the VM's primary IP (which is assigned to no pod). So we use docker's -p flag to map published ports to the main interface. This has the side effect of disallowing two pods from exposing the same port. (More discussion on this in [Issue #390](https://github.com/GoogleCloudPlatform/kubernetes/issues/390).)

We create a container to use for the pod network namespace -- a single loopback device and a single veth device. All the user's containers get their network namespaces from this pod networking container.

Docker allocates IP addresses from a bridge we create on each node, using its “container” networking mode.

1. Create a normal (in the networking sense) container which uses a minimal image and runs a command that blocks forever. This is not a user-defined container, and gets a special well-known name.
  - creates a new network namespace (netns) and loopback device
  - creates a new pair of veth devices and binds them to the netns
  - auto-assigns an IP from docker’s IP range

2. Create the user containers and specify the name of the pod infra container as their “POD” argument. Docker finds the PID of the command running in the pod infra container and attaches to the netns and ipcns of that PID.

### Other networking implementation examples
With the primary aim of providing IP-per-pod-model, other implementations exist to serve the purpose outside of GCE.
  - [OpenVSwitch with GRE/VxLAN](../ovs-networking.md)
  - [Flannel](https://github.com/coreos/flannel#flannel)

## Challenges and future work

### Docker API

Right now, docker inspect doesn't show the networking configuration of the containers, since they derive it from another container. That information should be exposed somehow.

### External IP assignment

We want to be able to assign IP addresses externally from Docker ([Docker issue #6743](https://github.com/dotcloud/docker/issues/6743)) so that we don't need to statically allocate fixed-size IP ranges to each node, so that IP addresses can be made stable across pod infra container restarts ([Docker issue #2801](https://github.com/dotcloud/docker/issues/2801)), and to facilitate pod migration. Right now, if the pod infra container dies, all the user containers must be stopped and restarted because the netns of the pod infra container will change on restart, and any subsequent user container restart will join that new netns, thereby not being able to see its peers. Additionally, a change in IP address would encounter DNS caching/TTL problems. External IP assignment would also simplify DNS support (see below).

### Naming, discovery, and load balancing

In addition to enabling self-registration with 3rd-party discovery mechanisms, we'd like to setup DDNS automatically ([Issue #146](https://github.com/GoogleCloudPlatform/kubernetes/issues/146)). hostname, $HOSTNAME, etc. should return a name for the pod ([Issue #298](https://github.com/GoogleCloudPlatform/kubernetes/issues/298)), and gethostbyname should be able to resolve names of other pods. Probably we need to set up a DNS resolver to do the latter ([Docker issue #2267](https://github.com/dotcloud/docker/issues/2267)), so that we don't need to keep /etc/hosts files up to date dynamically.

[Service](http://docs.k8s.io/services.md) endpoints are currently found through environment variables.  Both [Docker-links-compatible](https://docs.docker.com/userguide/dockerlinks/) variables and kubernetes-specific variables ({NAME}_SERVICE_HOST and {NAME}_SERVICE_BAR) are supported, and resolve to ports opened by the service proxy. We don't actually use [the Docker ambassador pattern](https://docs.docker.com/articles/ambassador_pattern_linking/) to link containers because we don't require applications to identify all clients at configuration time, yet.  While services today are managed by the service proxy, this is an implementation detail that applications should not rely on.  Clients should instead use the [service IP](http://docs.k8s.io/services.md) (which the above environment variables will resolve to).  However, a flat service namespace doesn't scale and environment variables don't permit dynamic updates, which complicates service deployment by imposing implicit ordering constraints.  We intend to register each service's IP in DNS, and for that to become the preferred resolution protocol.

We'd also like to accommodate other load-balancing solutions (e.g., HAProxy), non-load-balanced services ([Issue #260](https://github.com/GoogleCloudPlatform/kubernetes/issues/260)), and other types of groups (worker pools, etc.). Providing the ability to Watch a label selector applied to pod addresses would enable efficient monitoring of group membership, which could be directly consumed or synced with a discovery mechanism. Event hooks ([Issue #140](https://github.com/GoogleCloudPlatform/kubernetes/issues/140)) for join/leave events would probably make this even easier.

### External routability

We want traffic between containers to use the pod IP addresses across nodes. Say we have Node A with a container IP space of 10.244.1.0/24 and Node B with a container IP space of 10.244.2.0/24. And we have Container A1 at 10.244.1.1 and Container B1 at 10.244.2.1. We want Container A1 to talk to Container B1 directly with no NAT. B1 should see the "source" in the IP packets of 10.244.1.1 -- not the "primary" host IP for Node A. That means that we want to turn off NAT for traffic between containers (and also between VMs and containers).

We'd also like to make pods directly routable from the external internet. However, we can't yet support the extra container IPs that we've provisioned talking to the internet directly. So, we don't map external IPs to the container IPs. Instead, we solve that problem by having traffic that isn't to the internal network (! 10.0.0.0/8) get NATed through the primary host IP address so that it can get 1:1 NATed by the GCE networking when talking to the internet. Similarly, incoming traffic from the internet has to get NATed/proxied through the host IP.

So we end up with 3 cases:

1. Container -> Container or Container <-> VM. These should use 10. addresses directly and there should be no NAT.

2. Container -> Internet. These have to get mapped to the primary host IP so that GCE knows how to egress that traffic. There is actually 2 layers of NAT here: Container IP -> Internal Host IP -> External Host IP. The first level happens in the guest with IP tables and the second happens as part of GCE networking. The first one (Container IP -> internal host IP) does dynamic port allocation while the second maps ports 1:1.

3. Internet -> Container. This also has to go through the primary host IP and also has 2 levels of NAT, ideally. However, the path currently is a proxy with (External Host IP -> Internal Host IP -> Docker) -> (Docker -> Container IP). Once [issue #15](https://github.com/GoogleCloudPlatform/kubernetes/issues/15) is closed, it should be External Host IP -> Internal Host IP -> Container IP. But to get that second arrow we have to set up the port forwarding iptables rules per mapped port.

Another approach could be to create a new host interface alias for each pod, if we had a way to route an external IP to it. This would eliminate the scheduling constraints resulting from using the host's IP address.

### IPv6

IPv6 would be a nice option, also, but we can't depend on it yet. Docker support is in progress: [Docker issue #2974](https://github.com/dotcloud/docker/issues/2974), [Docker issue #6923](https://github.com/dotcloud/docker/issues/6923), [Docker issue #6975](https://github.com/dotcloud/docker/issues/6975). Additionally, direct ipv6 assignment to instances doesn't appear to be supported by major cloud providers (e.g., AWS EC2, GCE) yet. We'd happily take pull requests from people running Kubernetes on bare metal, though. :-)


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/networking.md?pixel)]()


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/release-0.19.0/docs/design/networking.md?pixel)]()
