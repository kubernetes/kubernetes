# memberlist [![GoDoc](https://godoc.org/github.com/hashicorp/memberlist?status.png)](https://godoc.org/github.com/hashicorp/memberlist)

memberlist is a [Go](http://www.golang.org) library that manages cluster
membership and member failure detection using a gossip based protocol.

The use cases for such a library are far-reaching: all distributed systems
require membership, and memberlist is a re-usable solution to managing
cluster membership and node failure detection.

memberlist is eventually consistent but converges quickly on average.
The speed at which it converges can be heavily tuned via various knobs
on the protocol. Node failures are detected and network partitions are partially
tolerated by attempting to communicate to potentially dead nodes through
multiple routes.

## Building

If you wish to build memberlist you'll need Go version 1.2+ installed.

Please check your installation with:

```
go version
```

## Usage

Memberlist is surprisingly simple to use. An example is shown below:

```go
/* Create the initial memberlist from a safe configuration.
   Please reference the godoc for other default config types.
   http://godoc.org/github.com/hashicorp/memberlist#Config
*/
list, err := memberlist.Create(memberlist.DefaultLocalConfig())
if err != nil {
	panic("Failed to create memberlist: " + err.Error())
}

// Join an existing cluster by specifying at least one known member.
n, err := list.Join([]string{"1.2.3.4"})
if err != nil {
	panic("Failed to join cluster: " + err.Error())
}

// Ask for members of the cluster
for _, member := range list.Members() {
	fmt.Printf("Member: %s %s\n", member.Name, member.Addr)
}

// Continue doing whatever you need, memberlist will maintain membership
// information in the background. Delegates can be used for receiving
// events when members join or leave.
```

The most difficult part of memberlist is configuring it since it has many
available knobs in order to tune state propagation delay and convergence times.
Memberlist provides a default configuration that offers a good starting point,
but errs on the side of caution, choosing values that are optimized for
higher convergence at the cost of higher bandwidth usage.

For complete documentation, see the associated [Godoc](http://godoc.org/github.com/hashicorp/memberlist).

## Protocol

memberlist is based on ["SWIM: Scalable Weakly-consistent Infection-style Process Group Membership Protocol"](http://www.cs.cornell.edu/~asdas/research/dsn02-swim.pdf),
with a few minor adaptations, mostly to increase propagation speed and
convergence rate.

A high level overview of the memberlist protocol (based on SWIM) is
described below, but for details please read the full
[SWIM paper](http://www.cs.cornell.edu/~asdas/research/dsn02-swim.pdf)
followed by the memberlist source. We welcome any questions related
to the protocol on our issue tracker.

### Protocol Description

memberlist begins by joining an existing cluster or starting a new
cluster. If starting a new cluster, additional nodes are expected to join
it. New nodes in an existing cluster must be given the address of at
least one existing member in order to join the cluster. The new member
does a full state sync with the existing member over TCP and begins gossiping its
existence to the cluster.

Gossip is done over UDP with a configurable but fixed fanout and interval.
This ensures that network usage is constant with regards to number of nodes, as opposed to
exponential growth that can occur with traditional heartbeat mechanisms.
Complete state exchanges with a random node are done periodically over
TCP, but much less often than gossip messages. This increases the likelihood
that the membership list converges properly since the full state is exchanged
and merged. The interval between full state exchanges is configurable or can
be disabled entirely.

Failure detection is done by periodic random probing using a configurable interval.
If the node fails to ack within a reasonable time (typically some multiple
of RTT), then an indirect probe as well as a direct TCP probe are attempted. An
indirect probe asks a configurable number of random nodes to probe the same node,
in case there are network issues causing our own node to fail the probe. The direct
TCP probe is used to help identify the common situation where networking is
misconfigured to allow TCP but not UDP. Without the TCP probe, a UDP-isolated node
would think all other nodes were suspect and could cause churn in the cluster when
it attempts a TCP-based state exchange with another node. It is not desirable to
operate with only TCP connectivity because convergence will be much slower, but it
is enabled so that memberlist can detect this situation and alert operators.

If both our probe, the indirect probes, and the direct TCP probe fail within a
configurable time, then the node is marked "suspicious" and this knowledge is
gossiped to the cluster. A suspicious node is still considered a member of
cluster. If the suspect member of the cluster does not dispute the suspicion
within a configurable period of time, the node is finally considered dead,
and this state is then gossiped to the cluster.

This is a brief and incomplete description of the protocol. For a better idea,
please read the
[SWIM paper](http://www.cs.cornell.edu/~asdas/research/dsn02-swim.pdf)
in its entirety, along with the memberlist source code.

### Changes from SWIM

As mentioned earlier, the memberlist protocol is based on SWIM but includes
minor changes, mostly to increase propagation speed and convergence rates.

The changes from SWIM are noted here:

* memberlist does a full state sync over TCP periodically. SWIM only propagates
  changes over gossip. While both eventually reach convergence, the full state
  sync increases the likelihood that nodes are fully converged more quickly,
  at the expense of more bandwidth usage. This feature can be totally disabled
  if you wish.

* memberlist has a dedicated gossip layer separate from the failure detection
  protocol. SWIM only piggybacks gossip messages on top of probe/ack messages.
  memberlist also piggybacks gossip messages on top of probe/ack messages, but
  also will periodically send out dedicated gossip messages on their own. This
  feature lets you have a higher gossip rate (for example once per 200ms)
  and a slower failure detection rate (such as once per second), resulting
  in overall faster convergence rates and data propagation speeds. This feature
  can be totally disabed as well, if you wish.

* memberlist stores around the state of dead nodes for a set amount of time,
  so that when full syncs are requested, the requester also receives information
  about dead nodes. Because SWIM doesn't do full syncs, SWIM deletes dead node
  state immediately upon learning that the node is dead. This change again helps
  the cluster converge more quickly.
