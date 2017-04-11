# Versioning

Goal: We want to be able to upgrade an individual peer in an etcd cluster to a newer version of etcd.
The process will take the form of individual followers upgrading to the latest version until the entire cluster is on the new version.

Immediate need: etcd is moving too fast to version the internal API right now.
But, we need to keep mixed version clusters from being started by a rolling upgrade process (e.g. the CoreOS developer alpha).

Longer term need: Having a mixed version cluster where all peers are not running the exact same version of etcd itself but are able to speak one version of the internal protocol.

Solution: The internal protocol needs to be versioned just as the client protocol is.
Initially during the 0.\*.\* series of etcd releases we won't allow mixed versions at all.

## Join Control

We will add a version field to the join command.
But, who decides whether a newly upgraded follower should be able to join a cluster?

### Leader Controlled

If the leader controls the version of followers joining the cluster then it compares its version to the version number presented by the follower in the JoinCommand and rejects the join if the number is less than the leader's version number.

Advantages

- Leader controls all cluster decisions still

Disadvantages

- Follower knows better what versions of the internal protocol it can talk than the leader


### Follower Controlled

A newly upgraded follower should be able to figure out the leaders internal version from a defined internal backwards compatible API endpoint and figure out if it can join the cluster.
If it cannot join the cluster then it simply exits.

Advantages

- The follower is running newer code and knows better if it can talk older protocols

Disadvantages

- This cluster decision isn't made by the leader

## Recommendation

To solve the immediate need and to plan for the future lets do the following:

- Add Version field to JoinCommand
- Have a joining follower read the Version field of the leader and if its own version doesn't match the leader then sleep for some random interval and retry later to see if the leader has upgraded.

# Research

## Zookeeper versioning

Zookeeper very recently added versioning into the protocol and it doesn't seem to have seen any use yet.
https://issues.apache.org/jira/browse/ZOOKEEPER-1633

## doozerd

doozerd stores the version number of the peers in the datastore for other clients to check, no decisions are made off of this number currently.
