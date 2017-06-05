etcdctl
========

`etcdctl` is a command line client for [etcd][etcd].
Make sure to set environment variable `ETCDCTL_API=3`. For etcdctl v2, please check [READMEv2][READMEv2].

## Commands

### PUT [options] \<key\> \<value\>

PUT assigns the specified value with the specified key. If key already holds a value, it is overwritten.

#### Options

- lease -- lease ID (in hexadecimal) to attach to the key.

#### Return value

##### Simple reply

- OK if PUT executed correctly. Exit code is zero.

- Error string if PUT failed. Exit code is non-zero.

##### JSON reply

The JSON encoding of the PUT [RPC response][etcdrpc].

##### Protobuf reply

The protobuf encoding of the PUT [RPC response][etcdrpc].

#### Examples

``` bash
./etcdctl PUT foo bar --lease=0x1234abcd
OK
./etcdctl range foo
bar
```

#### Notes

If \<value\> isn't given as command line argument, this command tries to read the value from standard input.

When \<value\> begins with '-', \<value\> is interpreted as a flag.
Insert '--' for workaround:

``` bash
./etcdctl put <key> -- <value>
./etcdctl put -- <key> <value>
```

### GET [options] \<key\> [range_end]

GET gets the key or a range of keys [key, range_end) if `range-end` is given.

#### Options

- hex -- print out key and value as hex encode string

- limit -- maximum number of results

- prefix -- get keys by matching prefix

- order -- order of results; ASCEND or DESCEND

- sort-by -- sort target; CREATE, KEY, MODIFY, VALUE, or VERSION

- rev -- specify the kv revision

TODO: add consistency, from, prefix

#### Return value

##### Simple reply

- \<key\>\n\<value\>\n\<next_key\>\n\<next_value\>...

- Error string if GET failed. Exit code is non-zero.

##### JSON reply

The JSON encoding of the [RPC response][etcdrpc] for the GET's Range request.

##### Protobuf reply

The protobuf encoding of the [RPC message][etcdrpc] for a key-value pair for each fetched key-value.

#### Examples

``` bash
./etcdctl get foo
foo
bar
```

#### Notes

If any key or value contains non-printable characters or control characters, the output in text format (e.g. simple reply) might be ambiguous.
Adding `--hex` to print key or value as hex encode string in text format can resolve this issue.

### DEL [options] \<key\> [range_end]

Removes the specified key or range of keys [key, range_end) if `range-end` is given.

#### Options

- prefix -- delete keys by matching prefix

TODO: --from

#### Return value

##### Simple reply

- The number of keys that were removed in decimal if DEL executed correctly. Exit code is zero.

- Error string if DEL failed. Exit code is non-zero.

##### JSON reply

The JSON encoding of the DeleteRange [RPC response][etcdrpc].

##### Protobuf reply

The protobuf encoding of the DeleteRange [RPC response][etcdrpc].

#### Examples

``` bash
./etcdctl put foo bar
OK
./etcdctl del foo
1
./etcdctl range foo
```

### TXN [options]

TXN reads multiple etcd requests from standard input and applies them as a single atomic transaction.
A transaction consists of list of conditions, a list of requests to apply if all the conditions are true, and a list of requests to apply if any condition is false.

#### Options

- hex -- print out keys and values as hex encoded string

- interactive -- input transaction with interactive prompting

#### Input Format
```ebnf
<Txn> ::= <CMP>* "\n" <THEN> "\n" <ELSE> "\n"
<CMP> ::= (<CMPCREATE>|<CMPMOD>|<CMPVAL>|<CMPVER>) "\n"
<CMPOP> ::= "<" | "=" | ">"
<CMPCREATE> := ("c"|"create")"("<KEY>")" <REVISION>
<CMPMOD> ::= ("m"|"mod")"("<KEY>")" <CMPOP> <REVISION>
<CMPVAL> ::= ("val"|"value")"("<KEY>")" <CMPOP> <VALUE>
<CMPVER> ::= ("ver"|"version")"("<KEY>")" <CMPOP> <VERSION>
<THEN> ::= <OP>*
<ELSE> ::= <OP>*
<OP> ::= ((see put, get, del etcdctl command syntax)) "\n"
<KEY> ::= (%q formatted string)
<VALUE> ::= (%q formatted string)
<REVISION> ::= "\""[0-9]+"\""
<VERSION> ::= "\""[0-9]+"\""
```

#### Return value

##### Simple reply

- SUCCESS if etcd processed the transaction success list, FAILURE if etcd processed the transaction failure list.

- Simple reply for each command executed request list, each separated by a blank line.

- Additional error string if TXN failed. Exit code is non-zero.

##### JSON reply

The JSON encoding of the Txn [RPC response][etcdrpc].

##### Protobuf reply

The protobuf encoding of the Txn [RPC response][etcdrpc].

#### Examples

txn in interactive mode:
``` bash
./etcdctl txn -i
mod("key1") > "0"

put key1 "overwrote-key1"

put key1 "created-key1"
put key2 "some extra key"

FAILURE

OK

OK
```

txn in non-interactive mode:
```
./etcdctl txn <<<'mod("key1") > "0"

put key1 "overwrote-key1"

put key1 "created-key1"
put key2 "some extra key"

'
FAILURE

OK

OK
````

### WATCH [options] [key or prefix] [range_end]

Watch watches events stream on keys or prefixes, [key or prefix, range_end) if `range-end` is given. The watch command runs until it encounters an error or is terminated by the user.  If range_end is given, it must be lexicographically greater than key or "\x00".

#### Options

- hex -- print out key and value as hex encode string

- interactive -- begins an interactive watch session

- prefix -- watch on a prefix if prefix is set.

- prev-kv -- get the previous key-value pair before the event happens.

- rev -- the revision to start watching. Specifying a revision is useful for observing past events.

#### Input Format

Input is only accepted for interactive mode.

```
watch [options] <key or prefix>\n
```

#### Return value

##### Simple reply

- \<event\>[\n\<old_key\>\n\<old_value\>]\n\<key\>\n\<value\>\n\<event\>\n\<next_key\>\n\<next_value\>\n...

- Additional error string if WATCH failed. Exit code is non-zero.

##### JSON reply

The JSON encoding of the [RPC message][storagerpc] for each received Event.

##### Protobuf reply

The protobuf encoding of the [RPC message][storagerpc] for each received Event.

#### Examples

##### Non-interactive

``` bash
./etcdctl watch foo
PUT
foo
bar
```

##### Interactive

``` bash
./etcdctl watch -i
watch foo
watch foo
PUT
foo
bar
PUT
foo
bar
```

### LEASE \<subcommand\>

LEASE provides commands for key lease management.

### LEASE GRANT \<ttl\>

LEASE GRANT creates a fresh lease with a server-selected time-to-live in seconds
greater than or equal to the requested TTL value.

#### Return value

- On success, prints a message with the granted lease ID.

- On failure, prints an error message and returns with a non-zero exit code.

#### Example

```bash
./etcdctl lease grant 10
lease 32695410dcc0ca06 granted with TTL(10s)
```

### LEASE REVOKE \<leaseID\>

LEASE REVOKE destroys a given lease, deleting all attached keys.

#### Return value

- On success, prints a message indicating the lease is revoked.

- On failure, prints an error message and returns with a non-zero exit code.

#### Example

```bash
./etcdctl lease revoke 32695410dcc0ca06
lease 32695410dcc0ca06 revoked
```

### LEASE KEEP-ALIVE \<leaseID\>

LEASE KEEP-ALIVE periodically refreshes a lease so it does not expire.

#### Return value

- On success, prints a message for every keep alive sent.

- On failure, returns a non-zero exit code if a keep-alive channel could not be established. Otherwise, prints a message indicating the lease is gone.

#### Example
```bash
/etcdctl lease keep-alive 32695410dcc0ca0
lease 32695410dcc0ca0 keepalived with TTL(100)
lease 32695410dcc0ca0 keepalived with TTL(100)
lease 32695410dcc0ca0 keepalived with TTL(100)
...
```


### MEMBER \<subcommand\>

MEMBER provides commands for managing etcd cluster membership.

### MEMBER ADD \<memberName\>

MEMBER ADD introduces a new member into the etcd cluster as a new peer.

#### Options

- peer-urls -- comma separated list of URLs to associate with the new member.

#### Return value

- On success, prints the member ID of the new member and the cluster ID.

- On failure, prints an error message and returns with a non-zero exit code.

#### Example

```bash
./etcdctl member add newMember --peer-urls=https://127.0.0.1:12345
Member 2be1eb8f84b7f63e added to cluster ef37ad9dc622a7c4
```


### MEMBER UPDATE \<memberID\>

MEMBER UPDATE sets the peer URLs for an existing member in the etcd cluster.

#### Options

- peer-urls -- comma separated list of URLs to associate with the updated member.

#### Return value

- On success, prints the member ID of the updated member and the cluster ID.

- On failure, prints an error message and returns with a non-zero exit code.

#### Example

```bash
./etcdctl member update 2be1eb8f84b7f63e --peer-urls=https://127.0.0.1:11112
Member 2be1eb8f84b7f63e updated in cluster ef37ad9dc622a7c4
```


### MEMBER REMOVE \<memberID\>

MEMBER REMOVE removes a member of an etcd cluster from participating in cluster consensus.

#### Return value

- On success, prints the member ID of the removed member and the cluster ID.

- On failure, prints an error message and returns with a non-zero exit code.

#### Example

```bash
./etcdctl member remove 2be1eb8f84b7f63e
Member 2be1eb8f84b7f63e removed from cluster ef37ad9dc622a7c4
```

### MEMBER LIST

MEMBER LIST prints the member details for all members associated with an etcd cluster.

#### Return value

##### Simple reply

On success, prints a humanized table of the member IDs, statuses, names, peer addresses, and client addresses. On failure, prints an error message and returns with a non-zero exit code.

##### JSON reply

On success, prints a JSON listing of the member IDs, statuses, names, peer addresses, and client addresses. On failure, prints an error message and returns with a non-zero exit code.

##### Protobuf reply

The protobuf encoding of the MEMBER LIST [RPC response][member_list_rpc].

#### Examples

```bash
./etcdctl member list
8211f1d0f64f3269, started, infra1, http://127.0.0.1:12380, http://127.0.0.1:2379
91bc3c398fb3c146, started, infra2, http://127.0.0.1:22380, http://127.0.0.1:22379
fd422379fda50e48, started, infra3, http://127.0.0.1:32380, http://127.0.0.1:32379
```

```bash
./etcdctl -w json member list
{"header":{"cluster_id":17237436991929493444,"member_id":9372538179322589801,"raft_term":2},"members":[{"ID":9372538179322589801,"name":"infra1","peerURLs":["http://127.0.0.1:12380"],"clientURLs":["http://127.0.0.1:2379"]},{"ID":10501334649042878790,"name":"infra2","peerURLs":["http://127.0.0.1:22380"],"clientURLs":["http://127.0.0.1:22379"]},{"ID":18249187646912138824,"name":"infra3","peerURLs":["http://127.0.0.1:32380"],"clientURLs":["http://127.0.0.1:32379"]}]}
```

```bash
./etcdctl -w table member list
+------------------+---------+--------+------------------------+------------------------+
|        ID        | STATUS  |  NAME  |       PEER ADDRS       |      CLIENT ADDRS      |
+------------------+---------+--------+------------------------+------------------------+
| 8211f1d0f64f3269 | started | infra1 | http://127.0.0.1:12380 | http://127.0.0.1:2379  |
| 91bc3c398fb3c146 | started | infra2 | http://127.0.0.1:22380 | http://127.0.0.1:22379 |
| fd422379fda50e48 | started | infra3 | http://127.0.0.1:32380 | http://127.0.0.1:32379 |
+------------------+---------+--------+------------------------+------------------------+
```

## Utility Commands

### ENDPOINT \<subcommand\>

ENDPOINT provides commands for querying individual endpoints.

### ENDPOINT HEALTH

ENDPOINT HEALTH checks the health of the list of endpoints with respect to cluster. An endpoint is unhealthy
when it cannot participate in consensus with the rest of the cluster.

#### Return value

- If an endpoint can participate in consensus, prints a message indicating the endpoint is healthy.

- If an endpoint fails to participate in consensus, prints a message indicating the endpoint is unhealthy.

#### Example

```bash
./etcdctl endpoint health
127.0.0.1:32379 is healthy: successfully committed proposal: took = 2.130877ms
127.0.0.1:2379 is healthy: successfully committed proposal: took = 2.095242ms
127.0.0.1:22379 is healthy: successfully committed proposal: took = 2.083263ms
```

### ENDPOINT STATUS

ENDPOINT STATUS queries the status of each endpoint in the given endpoint list.

#### Return value

##### Simple reply

On success, prints a humanized table of each endpoint URL, ID, version, database size, leadership status, raft term, and raft status. On failure, returns with a non-zero exit code.

##### JSON reply

On success, prints a line of JSON encoding each endpoint URL, ID, version, database size, leadership status, raft term, and raft status. On failure, returns with a non-zero exit code.

##### Protobuf reply

ENDPOINT STATUS does not support protobuf encoded output.

#### Examples

```bash
./etcdctl endpoint status
127.0.0.1:2379, 8211f1d0f64f3269, 3.0.0, 25 kB, false, 2, 63
127.0.0.1:22379, 91bc3c398fb3c146, 3.0.0, 25 kB, false, 2, 63
127.0.0.1:32379, fd422379fda50e48, 3.0.0, 25 kB, true, 2, 63
```

```bash
./etcdctl -w json endpoint status
[{"Endpoint":"127.0.0.1:2379","Status":{"header":{"cluster_id":17237436991929493444,"member_id":9372538179322589801,"revision":2,"raft_term":2},"version":"3.0.0","dbSize":24576,"leader":18249187646912138824,"raftIndex":32623,"raftTerm":2}},{"Endpoint":"127.0.0.1:22379","Status":{"header":{"cluster_id":17237436991929493444,"member_id":10501334649042878790,"revision":2,"raft_term":2},"version":"3.0.0","dbSize":24576,"leader":18249187646912138824,"raftIndex":32623,"raftTerm":2}},{"Endpoint":"127.0.0.1:32379","Status":{"header":{"cluster_id":17237436991929493444,"member_id":18249187646912138824,"revision":2,"raft_term":2},"version":"3.0.0","dbSize":24576,"leader":18249187646912138824,"raftIndex":32623,"raftTerm":2}}]
```

```bash
./etcdctl -w table endpoint status
+-----------------+------------------+---------+---------+-----------+-----------+------------+
|    ENDPOINT     |        ID        | VERSION | DB SIZE | IS LEADER | RAFT TERM | RAFT INDEX |
+-----------------+------------------+---------+---------+-----------+-----------+------------+
| 127.0.0.1:2379  | 8211f1d0f64f3269 |  3.0.0  | 25 kB   | false     |         2 |         52 |
| 127.0.0.1:22379 | 91bc3c398fb3c146 |  3.0.0  | 25 kB   | false     |         2 |         52 |
| 127.0.0.1:32379 | fd422379fda50e48 |  3.0.0  | 25 kB   | true      |         2 |         52 |
+-----------------+------------------+---------+---------+-----------+-----------+------------+
```

### LOCK \<lockname\>

LOCK acquires a distributed named mutex with a given name. Once the lock is acquired, it will be held until etcdctl is terminated.

#### Return value

- Once the lock is acquired, the result for the GET on the unique lock holder key is displayed.

- LOCK returns a zero exit code only if it is terminated by a signal and can release the lock.

#### Example
```bash
./etcdctl lock mylock
mylock/1234534535445


```

#### Notes

The lease length of a lock defaults to 60 seconds. If LOCK is abnormally terminated, lock progress may be delayed
by up to 60 seconds.


### ELECT [options] \<election-name\> [proposal]

ELECT participates on a named election. A node announces its candidacy in the election by providing
a proposal value. If a node wishes to observe the election, ELECT listens for new leaders values.
Whenever a leader is elected, its proposal is given as output.

#### Options

- listen -- observe the election

#### Return value

- If a candidate, ELECT displays the GET on the leader key once the node is elected election.

- If observing, ELECT streams the result for a GET on the leader key for the current election and all future elections.

- ELECT returns a zero exit code only if it is terminated by a signal and can revoke its candidacy or leadership, if any.

#### Example
```bash
./etcdctl elect myelection foo
myelection/1456952310051373265
foo

```

#### Notes

The lease length of a leader defaults to 60 seconds. If a candidate is abnormally terminated, election
progress may be delayed by up to 60 seconds.


### COMPACTION \<revision\>

COMPACTION discards all etcd event history prior to a given revision. Since etcd uses a multiversion concurrency control
model, it preserves all key updates as event history. When the event history up to some revision is no longer needed,
all superseded keys may be compacted away to reclaim storage space in the etcd backend database.

#### Return value

- On success, prints the compacted revision and returns a zero exit code.

- On failure, prints an error message and returns with a non-zero exit code.

#### Example
```bash
./etcdctl compaction 1234
compacted revision 1234
```

### DEFRAG

DEFRAG defragments the backend database file for a set of given endpoints. When an etcd member reclaims storage space
from deleted and compacted keys, the space is kept in a free list and the database file remains the same size. By defragmenting
the database, the etcd member releases this free space back to the file system.

#### Return value

- If successfully defragmented an endpoint, prints a message indicating success for that endpoint.

- If failed defragmenting an endpoint, prints a message indicating failure for that endpoint.

- DEFRAG returns a zero exit code only if it succeeded defragmenting all given endpoints.

#### Example
```bash
./etcdctl --endpoints=localhost:2379,badendpoint:2379 defrag
Finished defragmenting etcd member[localhost:2379]
Failed to defragment etcd member[badendpoint:2379] (grpc: timed out trying to connect)
```


### MAKE-MIRROR [options] \<destination\>

[make-mirror][mirror] mirrors a key prefix in an etcd cluster to a destination etcd cluster.

#### Options

- dest-cacert -- TLS certificate authority file for destination cluster

- dest-cert -- TLS certificate file for destination cluster

- dest-key -- TLS key file for destination cluster

- prefix -- The key-value prefix to mirror

#### Return value

Simple reply

- The approximate total number of keys transferred to the destination cluster, updated every 30 seconds.

- Error string if mirroring failed. Exit code is non-zero.

#### Examples

```
./etcdctl make-mirror mirror.example.com:2379
10
18
```

[mirror]: ./doc/mirror_maker.md


### SNAPSHOT \<subcommand\>

SNAPSHOT provides commands to restore a snapshot of a running etcd server into a fresh cluster.

### SNAPSHOT SAVE \<filename\>

SNAPSHOT SAVE writes a point-in-time snapshot of the etcd backend database to a file.

#### Return value

- On success, the backend snapshot is written to the given file path.

- Error string if snapshotting failed. Exit code is non-zero.

#### Example

Save a snapshot to "snapshot.db":
```
./etcdctl snapshot save snapshot.db
```


### SNAPSHOT RESTORE [options] \<filename\>

SNAPSHOT RESTORE creates an etcd data directory for an etcd cluster member from a backend database snapshot and a new cluster configuration. Restoring the snapshot into each member for a new cluster configuration will initialize a new etcd cluster preloaded by the snapshot data.

#### Options

The snapshot restore options closely resemble to those used in the `etcd` command for defining a cluster.

- data-dir -- Path to the data directory. Uses \<name\>.etcd if none given.

- initial-cluster -- The initial cluster configuration for the restored etcd cluster.

- initial-cluster-token -- Initial cluster token for the restored etcd cluster.

- initial-advertise-peer-urls -- List of peer URLs for the member being restored.

- name -- Human-readable name for the etcd cluster member being restored.

#### Return value

- On success, a new etcd data directory is initialized.

- Error string if the data directory could not be completely initialized. Exit code is non-zero.

#### Example

Save a snapshot, restore into a new 3 node cluster, and start the cluster:
```
./etcdctl snapshot save snapshot.db

# restore members
bin/etcdctl snapshot restore snapshot.db --initial-cluster-token etcd-cluster-1 --initial-advertise-peer-urls http://127.0.0.1:12380  --name sshot1 --initial-cluster 'sshot1=http://127.0.0.1:12380,sshot2=http://127.0.0.1:22380,sshot3=http://127.0.0.1:32380'
bin/etcdctl snapshot restore snapshot.db --initial-cluster-token etcd-cluster-1 --initial-advertise-peer-urls http://127.0.0.1:22380  --name sshot2 --initial-cluster 'sshot1=http://127.0.0.1:12380,sshot2=http://127.0.0.1:22380,sshot3=http://127.0.0.1:32380'
bin/etcdctl snapshot restore snapshot.db --initial-cluster-token etcd-cluster-1 --initial-advertise-peer-urls http://127.0.0.1:32380  --name sshot3 --initial-cluster 'sshot1=http://127.0.0.1:12380,sshot2=http://127.0.0.1:22380,sshot3=http://127.0.0.1:32380'

# launch members
bin/etcd --name sshot1 --listen-client-urls http://127.0.0.1:2379 --advertise-client-urls http://127.0.0.1:2379 --listen-peer-urls http://127.0.0.1:12380 &
bin/etcd --name sshot2 --listen-client-urls http://127.0.0.1:22379 --advertise-client-urls http://127.0.0.1:22379 --listen-peer-urls http://127.0.0.1:22380 &
bin/etcd --name sshot3 --listen-client-urls http://127.0.0.1:32379 --advertise-client-urls http://127.0.0.1:32379 --listen-peer-urls http://127.0.0.1:32380 &
```

### SNAPSHOT STATUS \<filename\>

SNAPSHOT STATUS lists information about a given backend database snapshot file.

#### Return value

##### Simple Reply

On success, prints a humanized table of the database hash, revision, total keys, and size. On failure, return with a non-zero exit code.

##### JSON reply

On success, prints a line of JSON encoding the database hash, revision, total keys, and size. On failure, return with a non-zero exit code.

##### Protobuf reply

SNAPSHOT STATUS does not support protobuf encoded output.

#### Examples
```bash
./etcdctl snapshot status file.db
cf1550fb, 3, 3, 25 kB
```

```bash
./etcdctl -write-out=json snapshot status file.db
{"hash":3474280699,"revision":3,"totalKey":3,"totalSize":24576}
```

```bash
./etcdctl -write-out=table snapshot status file.db
+----------+----------+------------+------------+
|   HASH   | REVISION | TOTAL KEYS | TOTAL SIZE |
+----------+----------+------------+------------+
| cf1550fb |        3 |          3 | 25 kB      |
+----------+----------+------------+------------+
```

### MIGRATE [options]

Migrate migrates keys in a v2 store to a mvcc store. Users should run migration command for all members in the cluster.

#### Options

- data-dir -- Path to the data directory

- wal-dir -- Path to the WAL directory

- transformer -- Path to the user-provided transformer program (default if not provided)

#### Return value

Simple reply

- Exit code is zero when migration is finished successfully.

- Error string if migration failed. Exit code is non-zero.

#### Default transformer

If user does not provide a transformer program, migrate command will use the default transformer. The default transformer transforms `storev2` formatted keys into `mvcc` formatted keys according to the following Go program:

```go
func transform(n *storev2.Node) *mvccpb.KeyValue {
	if n.Dir {
		return nil
	}
	kv := &mvccpb.KeyValue{
		Key:            []byte(n.Key),
		Value:          []byte(n.Value),
		CreateRevision: int64(n.CreatedIndex),
		ModRevision:    int64(n.ModifiedIndex),
		Version:        1,
	}
	return kv
}
```

#### User-provided transformer

Users can provide a customized 1:n transformer function that transforms a key from the v2 store to any number of keys in the mvcc store. The migration program writes JSON formatted [v2 store keys][v2key] to the transformer program's stdin, reads protobuf formatted [mvcc keys][v3key] back from the transformer program's stdout, and finishes migration by saving the transformed keys into the mvcc store.

The provided transformer should read until EOF and flush the stdout before exiting to ensure data integrity.

#### Example

```
./etcdctl --data-dir=/var/etcd --transformer=k8s-transformer
finished transforming keys
```

### AUTH \<enable or disable\>

`auth enable` activates authentication on an etcd cluster and `auth disable` deactivates. When authentication is enabled, etcd checks all requests for appropriate authorization.

#### Return value

##### Simple reply

- `Authentication Enabled`. Exit code is zero.

- Error string if AUTH failed. Exit code is non-zero.

#### Examples

``` bash
etcdctl user add root
etcdctl user grant-role root root
etcdctl user get root
etcdctl role add root
etcdctl role get root

etcdctl auth enable
# Authentication Enabled
```

### ROLE ADD \<role name\>

`role add` creates a role.

#### Return value

##### Simple reply

- `Role <role name> created`. Exit code is zero.

- Error string if failed. Exit code is non-zero.

#### Examples

``` bash
etcdctl --user=root:123 role add myrole
```

### ROLE DELETE \<role name\>

`role delete` deletes a role.

#### Return value

##### Simple reply

- `Role <role name> deleted`. Exit code is zero.

- Error string if failed. Exit code is non-zero.

#### Examples

``` bash
etcdctl --user=root:123 role delete myrole
# Role myrole deleted
```

### ROLE GET \<role name\>

`role get` lists detailed role information.

#### Return value

##### Simple reply

- Detailed role information. Exit code is zero.

- Error string if failed. Exit code is non-zero.

#### Examples

``` bash
etcdctl --user=root:123 role get myrole
# Role myrole
# KV Read:
# 	foo
# KV Write:
# 	foo
```

### ROLE GRANT-PERMISSION \<role name\> \<permission type\> \<key\> [endkey]

`role grant-permission` grants a key to a role.

#### Return value

##### Simple reply

- `Role <role name> updated`. Exit code is zero.

- Error string if failed. Exit code is non-zero.

#### Examples

``` bash
etcdctl --user=root:123 role grant-permission myrole readwrite foo
# Role myrole updated
```

### ROLE REVOKE-PERMISSION \<role name\> \<permission type\> \<key\> [endkey]

`role revoke-permission` revokes a key from a role.

#### Return value

##### Simple reply

- `Permission of key <key> is revoked from role <role name>`. Exit code is zero.

- Error string if failed. Exit code is non-zero.

#### Examples

``` bash
etcdctl --user=root:123 role revoke-permission myrole foo
# Permission of key foo is revoked from role myrole
```

### USER ADD \<user name\>

`user add` creates a user.

#### Return value

##### Simple reply

- `User <user name> created`. Exit code is zero.

- Error string if failed. Exit code is non-zero.

#### Examples

``` bash
etcdctl --user=root:123 user add myuser
# Password of myuser: 
# Type password of myuser again for confirmation: 
# User myuser created
```

### USER DELETE \<user name\>

`user delete` deletes a user.

#### Return value

##### Simple reply

- `User <user name> deleted`. Exit code is zero.

- Error string if failed. Exit code is non-zero.

#### Examples

``` bash
etcdctl --user=root:123 user delete myuser
# User myuser deleted
```

### USER GET \<user name\>

`user get` lists detailed user information.

#### Return value

##### Simple reply

- Detailed user information. Exit code is zero.

- Error string if failed. Exit code is non-zero.

#### Examples

``` bash
etcdctl --user=root:123 user get myuser
# User: myuser
# Roles:
```

### USER PASSWD \<user name\>

`user passwd` changes a user's password.

#### Options

- interactive -- if true, read password in interactive terminal

#### Return value

##### Simple reply

- `Password updated`. Exit code is zero.

- Error string if failed. Exit code is non-zero.

#### Examples

``` bash
etcdctl --user=root:123 user passwd myuser
# Password of myuser: 
# Type password of myuser again for confirmation: 
# Password updated
```

### USER GRANT-ROLE \<user name\> \<role name\>

`user grant-role` grants a role to a user

#### Return value

##### Simple reply

- `Role <role name> is granted to user <user name>`. Exit code is zero.

- Error string if failed. Exit code is non-zero.

#### Examples

``` bash
etcdctl --user=root:123 user grant-role userA roleA
# Role roleA is granted to user userA
```

### USER REVOKE-ROLE \<user name\> \<role name\>

`user revoke-role` revokes a role from a user

#### Return value

##### Simple reply

- `Role <role name> is revoked from user <user name>`. Exit code is zero.

- Error string if failed. Exit code is non-zero.

#### Examples

``` bash
etcdctl --user=root:123 user revoke-role userA roleA
# Role roleA is revoked from user userA
```

## Notes

- JSON encoding for keys and values uses base64 since they are byte strings.


[etcdrpc]: ../etcdserver/etcdserverpb/rpc.proto
[storagerpc]: ../mvcc/mvccpb/kv.proto
[member_list_rpc]: ../etcdserver/etcdserverpb/rpc.proto#L493-L497

## Compatibility Support

etcdctl is still in its early stage. We try out best to ensure fully compatible releases, however we might break compatibility to fix bugs or improve commands. If we intend to release a version of etcdctl with backward incompatibilities, we will provide notice prior to release and have instructions on how to upgrade.

### Input Compatibility

Input includes the command name, its flags, and its arguments. We ensure backward compatibility of the input of normal commands in non-interactive mode.

### Output Compatibility

Output includes output from etcdctl and its exit code. etcdctl provides `simple` output format by default.
We ensure compatibility for the `simple` output format of normal commands in non-interactive mode. Currently, we do not ensure
backward compatibility for `JSON` format and the format in non-interactive mode. Currently, we do not ensure backward compatibility of utility commands.

### TODO: compatibility with etcd server

[etcd]: https://github.com/coreos/etcd
[READMEv2]: READMEv2.md
[v2key]: ../store/node_extern.go#L28-L37
[v3key]: ../mvcc/mvccpb/kv.proto#L12-L29
