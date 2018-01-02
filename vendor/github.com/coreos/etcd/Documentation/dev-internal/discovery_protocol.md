# Discovery service protocol

Discovery service protocol helps new etcd member to discover all other members in cluster bootstrap phase using a shared discovery URL.

Discovery service protocol is _only_ used in cluster bootstrap phase, and cannot be used for runtime reconfiguration or cluster monitoring.

The protocol uses a new discovery token to bootstrap one _unique_ etcd cluster. Remember that one discovery token can represent only one etcd cluster. As long as discovery protocol on this token starts, even if it fails halfway, it must not be used to bootstrap another etcd cluster.

The rest of this article will walk through the discovery process with examples that correspond to a self-hosted discovery cluster. The public discovery service, discovery.etcd.io, functions the same way, but with a layer of polish to abstract away ugly URLs, generate UUIDs automatically, and provide some protections against excessive requests. At its core, the public discovery service still uses an etcd cluster as the data store as described in this document.

## Protocol workflow

The idea of discovery protocol is to use an internal etcd cluster to coordinate bootstrap of a new cluster. First, all new members interact with discovery service and help to generate the expected member list. Then each new member bootstraps its server using this list, which performs the same functionality as -initial-cluster flag.

In the following example workflow, we will list each step of protocol in curl format for ease of understanding.

By convention the etcd discovery protocol uses the key prefix `_etcd/registry`. If `http://example.com` hosts an etcd cluster for discovery service, a full URL to discovery keyspace will be `http://example.com/v2/keys/_etcd/registry`. We will use this as the URL prefix in the example.

### Creating a new discovery token

Generate a unique token that will identify the new cluster. This will be used as a unique prefix in discovery keyspace in the following steps. An easy way to do this is to use `uuidgen`:

```
UUID=$(uuidgen)
```

### Specifying the expected cluster size

The discovery token expects a cluster size that must be specified. The size is used by the discovery service to know when it has found all members that will initially form the cluster.

```
curl -X PUT http://example.com/v2/keys/_etcd/registry/${UUID}/_config/size -d value=${cluster_size}
```

Usually the cluster size is 3, 5 or 7. Check [optimal cluster size][cluster-size] for more details.

### Bringing up etcd processes

Given the discovery URL, use it as `-discovery` flag and bring up etcd processes. Every etcd process will follow this next few steps internally if given a `-discovery` flag.

### Registering itself

The first thing for etcd process is to register itself into the discovery URL as a member. This is done by creating member ID as a key in the discovery URL.

```
curl -X PUT http://example.com/v2/keys/_etcd/registry/${UUID}/${member_id}?prevExist=false -d value="${member_name}=${member_peer_url_1}&${member_name}=${member_peer_url_2}"
```

### Checking the status

It checks the expected cluster size and registration status in discovery URL, and decides what the next action is.

```
curl -X GET http://example.com/v2/keys/_etcd/registry/${UUID}/_config/size
curl -X GET http://example.com/v2/keys/_etcd/registry/${UUID}
```

If registered members are still not enough, it will wait for left members to appear.

If the number of registered members is bigger than the expected size N, it treats the first N registered members as the member list for the cluster. If the member itself is in the member list, the discovery procedure succeeds and it fetches all peers through the member list. If it is not in the member list, the discovery procedure finishes with the failure that the cluster has been full.

In etcd implementation, the member may check the cluster status even before registering itself. So it could fail quickly if the cluster has been full.

### Waiting for all members

The wait process is described in detail in the [etcd API documentation][api].

```
curl -X GET http://example.com/v2/keys/_etcd/registry/${UUID}?wait=true&waitIndex=${current_etcd_index}
```

It keeps waiting until finding all members.

## Public discovery service

CoreOS Inc. hosts a public discovery service at https://discovery.etcd.io/ , which provides some nice features for ease of use.

### Mask key prefix

Public discovery service will redirect `https://discovery.etcd.io/${UUID}` to etcd cluster behind for the key at `/v2/keys/_etcd/registry`. It masks register key prefix for short and readable discovery url.

### Get new token

```
GET /new

Sent query:
	size=${cluster_size}
Possible status codes:
	200 OK
	400 Bad Request
200 Body:
	generated discovery url
```

The generation process in the service follows the steps from [Creating a New Discovery Token][new-discovery-token] to [Specifying the Expected Cluster Size][expected-cluster-size].

### Check discovery status

```
GET /${UUID}
```

The status for this discovery token, including the machines that have been registered, can be checked by requesting the value of the UUID.

### Open-source repository

The repository is located at https://github.com/coreos/discovery.etcd.io. It could be used to build a custom discovery service.

[api]: ../v2/api.md#waiting-for-a-change
[cluster-size]: ../v2/admin_guide.md#optimal-cluster-size
[expected-cluster-size]: #specifying-the-expected-cluster-size
[new-discovery-token]: #creating-a-new-discovery-token
