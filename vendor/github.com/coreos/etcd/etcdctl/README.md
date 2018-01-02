etcdctl
========

`etcdctl` is a command line client for [etcd][etcd].
Make sure to set environment variable `ETCDCTL_API=3`. For etcdctl v2, please check [READMEv2][READMEv2].

Global flags (e.g., `dial-timeout`, `--cacert`, `--cert`, `--key`) can be set with environment variables:

```
ETCDCTL_DIAL_TIMEOUT=3s
ETCDCTL_CACERT=/tmp/ca.pem
ETCDCTL_CERT=/tmp/cert.pem
ETCDCTL_KEY=/tmp/key.pem
```

Prefix flag strings with `ETCDCTL_`, convert all letters to upper-case, and replace dash(`-`) with underscore(`_`).

## Key-value commands

### PUT [options] \<key\> \<value\>

PUT assigns the specified value with the specified key. If key already holds a value, it is overwritten.

RPC: Put

#### Options

- lease -- lease ID (in hexadecimal) to attach to the key.

- prev-kv -- return the previous key-value pair before modification.

#### Output

`OK`

#### Examples

```bash
./etcdctl put foo bar --lease=1234abcd
# OK
./etcdctl get foo
# foo
# bar
```

```bash
./etcdctl put foo bar1 --prev-kv
# OK
# foo
# bar
./etcdctl get foo
# foo
# bar1
```

#### Remarks

If \<value\> isn't given as command line argument, this command tries to read the value from standard input.

When \<value\> begins with '-', \<value\> is interpreted as a flag.
Insert '--' for workaround:

```bash
./etcdctl put <key> -- <value>
./etcdctl put -- <key> <value>
```

### GET [options] \<key\> [range_end]

GET gets the key or a range of keys [key, range_end) if `range-end` is given.

RPC: Range

#### Options

- hex -- print out key and value as hex encode string

- limit -- maximum number of results

- prefix -- get keys by matching prefix

- order -- order of results; ASCEND or DESCEND

- sort-by -- sort target; CREATE, KEY, MODIFY, VALUE, or VERSION

- rev -- specify the kv revision

- print-value-only -- print only value when used with write-out=simple

- consistency -- Linearizable(l) or Serializable(s)

- from-key -- Get keys that are greater than or equal to the given key using byte compare

- keys-only -- Get only the keys

#### Output

\<key\>\n\<value\>\n\<next_key\>\n\<next_value\>...

#### Examples

```bash
./etcdctl put foo bar
# OK
./etcdctl put foo1 bar1
# OK
./etcdctl put foo2 bar2
# OK
./etcdctl put foo3 bar3
# OK
./etcdctl get foo
# foo
# bar
./etcdctl get --from-key foo1
# foo1
# bar1
# foo2
# bar2
# foo3
# bar3
./etcdctl get foo1 foo3
# foo1
# bar1
# foo2
# bar2
```

#### Remarks

If any key or value contains non-printable characters or control characters, simple formatted output can be ambiguous due to new lines. To resolve this issue, set `--hex` to hex encode all strings.

### DEL [options] \<key\> [range_end]

Removes the specified key or range of keys [key, range_end) if `range-end` is given.

RPC: DeleteRange

#### Options

- prefix -- delete keys by matching prefix

- prev-kv -- return deleted key-value pairs

- from-key -- delete keys that are greater than or equal to the given key using byte compare

#### Output

Prints the number of keys that were removed in decimal if DEL succeeded.

#### Examples

```bash
./etcdctl put foo bar
# OK
./etcdctl del foo
# 1
./etcdctl get foo
```

```bash
./etcdctl put key val
# OK
./etcdctl del --prev-kv key
# 1
# key
# val
./etcdctl get key
```

```bash
./etcdctl put a 123
# OK
./etcdctl put b 456
# OK
./etcdctl put z 789
# OK
./etcdctl del --from-key a
# 3
./etcdctl get --from-key a
```

```bash
./etcdctl put zoo val
# OK
./etcdctl put zoo1 val1
# OK
./etcdctl put zoo2 val2
# OK
./etcdctl del --prefix zoo
# 3
./etcdctl get zoo2
```

### TXN [options]

TXN reads multiple etcd requests from standard input and applies them as a single atomic transaction.
A transaction consists of list of conditions, a list of requests to apply if all the conditions are true, and a list of requests to apply if any condition is false.

RPC: Txn

#### Options

- hex -- print out keys and values as hex encoded strings.

- interactive -- input transaction with interactive prompting.

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

#### Output

`SUCCESS` if etcd processed the transaction success list, `FAILURE` if etcd processed the transaction failure list. Prints the output for each command in the executed request list, each separated by a blank line.

#### Examples

txn in interactive mode:
```bash
./etcdctl txn -i
# compares:
mod("key1") > "0"

# success requests (get, put, delete):
put key1 "overwrote-key1"

# failure requests (get, put, delete):
put key1 "created-key1"
put key2 "some extra key"

# FAILURE

# OK

# OK
```

txn in non-interactive mode:
```bash
./etcdctl txn <<<'mod("key1") > "0"

put key1 "overwrote-key1"

put key1 "created-key1"
put key2 "some extra key"

'

# FAILURE

# OK

# OK
```

### COMPACTION [options] \<revision\>

COMPACTION discards all etcd event history prior to a given revision. Since etcd uses a multiversion concurrency control
model, it preserves all key updates as event history. When the event history up to some revision is no longer needed,
all superseded keys may be compacted away to reclaim storage space in the etcd backend database.

RPC: Compact

#### Options

- physical -- 'true' to wait for compaction to physically remove all old revisions

#### Output

Prints the compacted revision.

#### Example
```bash
./etcdctl compaction 1234
# compacted revision 1234
```

### WATCH [options] [key or prefix] [range_end]

Watch watches events stream on keys or prefixes, [key or prefix, range_end) if `range-end` is given. The watch command runs until it encounters an error or is terminated by the user.  If range_end is given, it must be lexicographically greater than key or "\x00".

RPC: Watch

#### Options

- hex -- print out key and value as hex encode string

- interactive -- begins an interactive watch session

- prefix -- watch on a prefix if prefix is set.

- prev-kv -- get the previous key-value pair before the event happens.

- rev -- the revision to start watching. Specifying a revision is useful for observing past events.

#### Input format

Input is only accepted for interactive mode.

```
watch [options] <key or prefix>\n
```

#### Output

\<event\>[\n\<old_key\>\n\<old_value\>]\n\<key\>\n\<value\>\n\<event\>\n\<next_key\>\n\<next_value\>\n...

#### Examples

##### Non-interactive

```bash
./etcdctl watch foo
# PUT
# foo
# bar
```

##### Interactive

```bash
./etcdctl watch -i
watch foo
watch foo
# PUT
# foo
# bar
# PUT
# foo
# bar
```

### LEASE \<subcommand\>

LEASE provides commands for key lease management.

### LEASE GRANT \<ttl\>

LEASE GRANT creates a fresh lease with a server-selected time-to-live in seconds
greater than or equal to the requested TTL value.

RPC: LeaseGrant

#### Output

Prints a message with the granted lease ID.

#### Example

```bash
./etcdctl lease grant 10
# lease 32695410dcc0ca06 granted with TTL(10s)
```

### LEASE REVOKE \<leaseID\>

LEASE REVOKE destroys a given lease, deleting all attached keys.

RPC: LeaseRevoke

#### Output

Prints a message indicating the lease is revoked.

#### Example

```bash
./etcdctl lease revoke 32695410dcc0ca06
# lease 32695410dcc0ca06 revoked
```

### LEASE TIMETOLIVE \<leaseID\> [options]

LEASE TIMETOLIVE retrieves the lease information with the given lease ID.

RPC: LeaseTimeToLive

#### Options

- keys -- Get keys attached to this lease

#### Output

Prints lease information.

#### Example

```bash
./etcdctl lease grant 500
# lease 2d8257079fa1bc0c granted with TTL(500s)

./etcdctl put foo1 bar --lease=2d8257079fa1bc0c
# OK

./etcdctl put foo2 bar --lease=2d8257079fa1bc0c
# OK

./etcdctl lease timetolive 2d8257079fa1bc0c
# lease 2d8257079fa1bc0c granted with TTL(500s), remaining(481s)

./etcdctl lease timetolive 2d8257079fa1bc0c --keys
# lease 2d8257079fa1bc0c granted with TTL(500s), remaining(472s), attached keys([foo2 foo1])

./etcdctl lease timetolive 2d8257079fa1bc0c --write-out=json
# {"cluster_id":17186838941855831277,"member_id":4845372305070271874,"revision":3,"raft_term":2,"id":3279279168933706764,"ttl":465,"granted-ttl":500,"keys":null}

./etcdctl lease timetolive 2d8257079fa1bc0c --write-out=json --keys
# {"cluster_id":17186838941855831277,"member_id":4845372305070271874,"revision":3,"raft_term":2,"id":3279279168933706764,"ttl":459,"granted-ttl":500,"keys":["Zm9vMQ==","Zm9vMg=="]}
```

### LEASE KEEP-ALIVE \<leaseID\>

LEASE KEEP-ALIVE periodically refreshes a lease so it does not expire.

RPC: LeaseKeepAlive

#### Output

Prints a message for every keep alive sent or prints a message indicating the lease is gone.

#### Example
```bash
./etcdctl lease keep-alive 32695410dcc0ca0
# lease 32695410dcc0ca0 keepalived with TTL(100)
# lease 32695410dcc0ca0 keepalived with TTL(100)
# lease 32695410dcc0ca0 keepalived with TTL(100)
...
```

## Cluster maintenance commands

### MEMBER \<subcommand\>

MEMBER provides commands for managing etcd cluster membership.

### MEMBER ADD \<memberName\> [options]

MEMBER ADD introduces a new member into the etcd cluster as a new peer.

RPC: MemberAdd

#### Options

- peer-urls -- comma separated list of URLs to associate with the new member.

#### Output

Prints the member ID of the new member and the cluster ID.

#### Example

```bash
./etcdctl member add newMember --peer-urls=https://127.0.0.1:12345

Member ced000fda4d05edf added to cluster 8c4281cc65c7b112

ETCD_NAME="newMember"
ETCD_INITIAL_CLUSTER="newMember=https://127.0.0.1:12345,default=http://10.0.0.30:2380"
ETCD_INITIAL_CLUSTER_STATE="existing"
```

### MEMBER UPDATE \<memberID\> [options]

MEMBER UPDATE sets the peer URLs for an existing member in the etcd cluster.

RPC: MemberUpdate

#### Options

- peer-urls -- comma separated list of URLs to associate with the updated member.

#### Output

Prints the member ID of the updated member and the cluster ID.

#### Example

```bash
./etcdctl member update 2be1eb8f84b7f63e --peer-urls=https://127.0.0.1:11112
# Member 2be1eb8f84b7f63e updated in cluster ef37ad9dc622a7c4
```

### MEMBER REMOVE \<memberID\>

MEMBER REMOVE removes a member of an etcd cluster from participating in cluster consensus.

RPC: MemberRemove

#### Output

Prints the member ID of the removed member and the cluster ID.

#### Example

```bash
./etcdctl member remove 2be1eb8f84b7f63e
# Member 2be1eb8f84b7f63e removed from cluster ef37ad9dc622a7c4
```

### MEMBER LIST

MEMBER LIST prints the member details for all members associated with an etcd cluster.

RPC: [MemberList][member_list_rpc].

#### Output

Prints a humanized table of the member IDs, statuses, names, peer addresses, and client addresses.

#### Examples

```bash
./etcdctl member list
# 8211f1d0f64f3269, started, infra1, http://127.0.0.1:12380, http://127.0.0.1:2379
# 91bc3c398fb3c146, started, infra2, http://127.0.0.1:22380, http://127.0.0.1:22379
# fd422379fda50e48, started, infra3, http://127.0.0.1:32380, http://127.0.0.1:32379
```

```bash
./etcdctl -w json member list
# {"header":{"cluster_id":17237436991929493444,"member_id":9372538179322589801,"raft_term":2},"members":[{"ID":9372538179322589801,"name":"infra1","peerURLs":["http://127.0.0.1:12380"],"clientURLs":["http://127.0.0.1:2379"]},{"ID":10501334649042878790,"name":"infra2","peerURLs":["http://127.0.0.1:22380"],"clientURLs":["http://127.0.0.1:22379"]},{"ID":18249187646912138824,"name":"infra3","peerURLs":["http://127.0.0.1:32380"],"clientURLs":["http://127.0.0.1:32379"]}]}
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

### ENDPOINT \<subcommand\>

ENDPOINT provides commands for querying individual endpoints.

### ENDPOINT HEALTH

ENDPOINT HEALTH checks the health of the list of endpoints with respect to cluster. An endpoint is unhealthy
when it cannot participate in consensus with the rest of the cluster.

#### Output

If an endpoint can participate in consensus, prints a message indicating the endpoint is healthy. If an endpoint fails to participate in consensus, prints a message indicating the endpoint is unhealthy.

#### Example

```bash
./etcdctl endpoint health
# 127.0.0.1:32379 is healthy: successfully committed proposal: took = 2.130877ms
# 127.0.0.1:2379 is healthy: successfully committed proposal: took = 2.095242ms
# 127.0.0.1:22379 is healthy: successfully committed proposal: took = 2.083263ms
```

### ENDPOINT STATUS

ENDPOINT STATUS queries the status of each endpoint in the given endpoint list.

#### Output

##### Simple format

Prints a humanized table of each endpoint URL, ID, version, database size, leadership status, raft term, and raft status.

##### JSON format

Prints a line of JSON encoding each endpoint URL, ID, version, database size, leadership status, raft term, and raft status.

#### Examples

```bash
./etcdctl endpoint status
# 127.0.0.1:2379, 8211f1d0f64f3269, 3.0.0, 25 kB, false, 2, 63
# 127.0.0.1:22379, 91bc3c398fb3c146, 3.0.0, 25 kB, false, 2, 63
# 127.0.0.1:32379, fd422379fda50e48, 3.0.0, 25 kB, true, 2, 63
```

```bash
./etcdctl -w json endpoint status
# [{"Endpoint":"127.0.0.1:2379","Status":{"header":{"cluster_id":17237436991929493444,"member_id":9372538179322589801,"revision":2,"raft_term":2},"version":"3.0.0","dbSize":24576,"leader":18249187646912138824,"raftIndex":32623,"raftTerm":2}},{"Endpoint":"127.0.0.1:22379","Status":{"header":{"cluster_id":17237436991929493444,"member_id":10501334649042878790,"revision":2,"raft_term":2},"version":"3.0.0","dbSize":24576,"leader":18249187646912138824,"raftIndex":32623,"raftTerm":2}},{"Endpoint":"127.0.0.1:32379","Status":{"header":{"cluster_id":17237436991929493444,"member_id":18249187646912138824,"revision":2,"raft_term":2},"version":"3.0.0","dbSize":24576,"leader":18249187646912138824,"raftIndex":32623,"raftTerm":2}}]
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

### ALARM \<subcommand\>

Provides alarm related commands

### ALARM DISARM

`alarm disarm` Disarms all alarms

RPC: Alarm

#### Output

`alarm:<alarm type>` if alarm is present and disarmed.

#### Examples

```bash
./etcdctl alarm disarm
```

If NOSPACE alarm is present:

```bash
./etcdctl alarm disarm
# alarm:NOSPACE
```

### ALARM LIST

`alarm list` lists all alarms.

RPC: Alarm

#### Output

`alarm:<alarm type>` if alarm is present, empty string if no alarms present.

#### Examples

```bash
./etcdctl alarm list
```

If NOSPACE alarm is present:

```bash
./etcdctl alarm list
# alarm:NOSPACE
```

### DEFRAG

DEFRAG defragments the backend database file for a set of given endpoints. When an etcd member reclaims storage space
from deleted and compacted keys, the space is kept in a free list and the database file remains the same size. By defragmenting
the database, the etcd member releases this free space back to the file system.

#### Output

For each endpoints, prints a message indicating whether the endpoint was successfully defragmented.

#### Example

```bash
./etcdctl --endpoints=localhost:2379,badendpoint:2379 defrag
# Finished defragmenting etcd member[localhost:2379]
# Failed to defragment etcd member[badendpoint:2379] (grpc: timed out trying to connect)
```

#### Remarks

DEFRAG returns a zero exit code only if it succeeded defragmenting all given endpoints.

### SNAPSHOT \<subcommand\>

SNAPSHOT provides commands to restore a snapshot of a running etcd server into a fresh cluster.

### SNAPSHOT SAVE \<filename\>

SNAPSHOT SAVE writes a point-in-time snapshot of the etcd backend database to a file.

#### Output

The backend snapshot is written to the given file path.

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

- skip-hash-check -- Ignore snapshot integrity hash value (required if copied from data directory)

#### Output

A new etcd data directory initialized with the snapshot.

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

#### Output

##### Simple format

Prints a humanized table of the database hash, revision, total keys, and size.

##### JSON format

Prints a line of JSON encoding the database hash, revision, total keys, and size.

#### Examples
```bash
./etcdctl snapshot status file.db
# cf1550fb, 3, 3, 25 kB
```

```bash
./etcdctl -write-out=json snapshot status file.db
# {"hash":3474280699,"revision":3,"totalKey":3,"totalSize":24576}
```

```bash
./etcdctl -write-out=table snapshot status file.db
+----------+----------+------------+------------+
|   HASH   | REVISION | TOTAL KEYS | TOTAL SIZE |
+----------+----------+------------+------------+
| cf1550fb |        3 |          3 | 25 kB      |
+----------+----------+------------+------------+
```

## Concurrency commands

### LOCK \<lockname\>

LOCK acquires a distributed named mutex with a given name. Once the lock is acquired, it will be held until etcdctl is terminated.

#### Output

Once the lock is acquired, the result for the GET on the unique lock holder key is displayed.

#### Example

```bash
./etcdctl lock mylock
# mylock/1234534535445
```

#### Remarks

LOCK returns a zero exit code only if it is terminated by a signal and releases the lock.

If LOCK is abnormally terminated or fails to contact the cluster to release the lock, the lock will remain held until the lease expires. Progress may be delayed by up to the default lease length of 60 seconds.

### ELECT [options] \<election-name\> [proposal]

ELECT participates on a named election. A node announces its candidacy in the election by providing
a proposal value. If a node wishes to observe the election, ELECT listens for new leaders values.
Whenever a leader is elected, its proposal is given as output.

#### Options

- listen -- observe the election.

#### Output

- If a candidate, ELECT displays the GET on the leader key once the node is elected election.

- If observing, ELECT streams the result for a GET on the leader key for the current election and all future elections.

#### Example

```bash
./etcdctl elect myelection foo
# myelection/1456952310051373265
# foo
```

#### Remarks

ELECT returns a zero exit code only if it is terminated by a signal and can revoke its candidacy or leadership, if any.

If a candidate is abnormally terminated, election rogress may be delayed by up to the default lease length of 60 seconds.

## Authentication commands

### AUTH \<enable or disable\>

`auth enable` activates authentication on an etcd cluster and `auth disable` deactivates. When authentication is enabled, etcd checks all requests for appropriate authorization.

RPC: AuthEnable/AuthDisable

#### Output

`Authentication Enabled`.

#### Examples

```bash
./etcdctl user add root
# Password of root:#type password for root
# Type password of root again for confirmation:#re-type password for root
# User root created
./etcdctl user grant-role root root
# Role root is granted to user root
./etcdctl user get root
# User: root
# Roles: root
./etcdctl role add root
# Role root created
./etcdctl role get root
# Role root
# KV Read:
# KV Write:
./etcdctl auth enable
# Authentication Enabled
```

### ROLE \<subcommand\>

ROLE is used to specify differnt roles which can be assigned to etcd user(s).

### ROLE ADD \<role name\>

`role add` creates a role.

RPC: RoleAdd

#### Output

`Role <role name> created`.

#### Examples

```bash
./etcdctl --user=root:123 role add myrole
# Role myrole created
```

### ROLE GET \<role name\>

`role get` lists detailed role information.

RPC: RoleGet

#### Output

Detailed role information.

#### Examples

```bash
./etcdctl --user=root:123 role get myrole
# Role myrole
# KV Read:
# foo
# KV Write:
# foo
```

### ROLE DELETE \<role name\>

`role delete` deletes a role.

RPC: RoleDelete

#### Output

`Role <role name> deleted`.

#### Examples

```bash
./etcdctl --user=root:123 role delete myrole
# Role myrole deleted
```

### ROLE LIST \<role name\>

`role list` lists all roles in etcd.

RPC: RoleList

#### Output

A role per line.

#### Examples

```bash
./etcdctl --user=root:123 role list
# roleA
# roleB
# myrole
```

### ROLE GRANT-PERMISSION [options] \<role name\> \<permission type\> \<key\> [endkey]

`role grant-permission` grants a key to a role.

RPC: RoleGrantPermission

#### Options

- prefix -- grant a prefix permission

#### Ouptut

`Role <role name> updated`.

#### Examples

```bash
./etcdctl --user=root:123 role grant-permission myrole readwrite foo
# Role myrole updated
```

### ROLE REVOKE-PERMISSION \<role name\> \<permission type\> \<key\> [endkey]

`role revoke-permission` revokes a key from a role.

RPC: RoleRevokePermission

#### Output

`Permission of key <key> is revoked from role <role name>` for single key. `Permission of range [<key>, <endkey>) is revoked from role <role name>` for a key range. Exit code is zero.

#### Examples

```bash
./etcdctl --user=root:123 role revoke-permission myrole foo
# Permission of key foo is revoked from role myrole
```

### USER \<subcommand\>

USER provides commands for managing users of etcd.

### USER ADD \<user name or user:password\> [options]

`user add` creates a user.

RPC: UserAdd

#### Options

- interactive -- Read password from stdin instead of interactive terminal

#### Output

`User <user name> created`.

#### Examples

```bash
./etcdctl --user=root:123 user add myuser
# Password of myuser: #type password for my user
# Type password of myuser again for confirmation:#re-type password for my user
# User myuser created
```

### USER GET \<user name\> [options]

`user get` lists detailed user information.

RPC: UserGet

#### Options

- detail -- Show permissions of roles granted to the user

#### Output

Detailed user information.

#### Examples

```bash
./etcdctl --user=root:123 user get myuser
# User: myuser
# Roles:
```

### USER DELETE \<user name\>

`user delete` deletes a user.

RPC: UserDelete

#### Output

`User <user name> deleted`.

#### Examples

```bash
./etcdctl --user=root:123 user delete myuser
# User myuser deleted
```

### USER LIST

`user list` lists detailed user information.

RPC: UserList

#### Output

- List of users, one per line.

#### Examples

```bash
./etcdctl --user=root:123 user list
# user1
# user2
# myuser
```

### USER PASSWD \<user name\> [options]

`user passwd` changes a user's password.

RPC: UserChangePassword

#### Options

- interactive -- if true, read password in interactive terminal

#### Output

`Password updated`.

#### Examples

```bash
./etcdctl --user=root:123 user passwd myuser
# Password of myuser: #type new password for my user
# Type password of myuser again for confirmation: #re-type the new password for my user
# Password updated
```

### USER GRANT-ROLE \<user name\> \<role name\>

`user grant-role` grants a role to a user

RPC: UserGrantRole

#### Output

`Role <role name> is granted to user <user name>`.

#### Examples

```bash
./etcdctl --user=root:123 user grant-role userA roleA
# Role roleA is granted to user userA
```

### USER REVOKE-ROLE \<user name\> \<role name\>

`user revoke-role` revokes a role from a user

RPC: UserRevokeRole

#### Output

`Role <role name> is revoked from user <user name>`.

#### Examples

```bash
./etcdctl --user=root:123 user revoke-role userA roleA
# Role roleA is revoked from user userA
```

## Utility commands

### MAKE-MIRROR [options] \<destination\>

[make-mirror][mirror] mirrors a key prefix in an etcd cluster to a destination etcd cluster.

#### Options

- dest-cacert -- TLS certificate authority file for destination cluster

- dest-cert -- TLS certificate file for destination cluster

- dest-key -- TLS key file for destination cluster

- prefix -- The key-value prefix to mirror

- dest-prefix -- The destination prefix to mirror a prefix to a different prefix in the destination cluster

- no-dest-prefix -- Mirror key-values to the root of the destination cluster

- dest-insecure-transport -- Disable transport security for client connections

#### Output

The approximate total number of keys transferred to the destination cluster, updated every 30 seconds.

#### Examples

```
./etcdctl make-mirror mirror.example.com:2379
# 10
# 18
```

[mirror]: ./doc/mirror_maker.md

### MIGRATE [options]

Migrates keys in a v2 store to a v3 mvcc store. Users should run migration command for all members in the cluster.

#### Options

- data-dir -- Path to the data directory

- wal-dir -- Path to the WAL directory

- transformer -- Path to the user-provided transformer program (default if not provided)

#### Output

No output on success.

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
# finished transforming keys
```

### VERSION

Prints the version of etcdctl.

#### Output

Prints etcd version and API version.

#### Examples

```bash
./etcdctl version
# etcdctl version: 3.1.0-alpha.0+git
# API version: 3.1
```

## Exit codes

For all commands, a successful execution return a zero exit code. All failures will return non-zero exit codes.

## Output formats

All commands accept an output format by setting `-w` or `--write-out`. All commands default to the "simple" output format, which is meant to be human-readable. The simple format is listed in each command's `Output` description since it is customized for each command. If a command has a corresponding RPC, it will respect all output formats.

If a command fails, returning a non-zero exit code, an error string will be written to standard error regardless of output format.

### Simple

A format meant to be easy to parse and human-readable. Specific to each command.

### JSON

The JSON encoding of the command's [RPC response][etcdrpc]. Since etcd's RPCs use byte strings, the JSON output will encode keys and values in base64.

Some commands without an RPC also support JSON; see the command's `Output` description.

### Protobuf

The protobuf encoding of the command's [RPC response][etcdrpc]. If an RPC is streaming, the stream messages will be concetenated. If an RPC is not given for a command, the protobuf output is not defined.

### Fields

An output format similar to JSON but meant to parse with coreutils. For an integer field named `Field`, it writes a line in the format `"Field" : %d` where `%d` is go's integer formatting. For byte array fields, it writes `"Field" : %q` where `%q` is go's quoted string formatting (e.g., `[]byte{'a', '\n'}` is written as `"a\n"`).

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
[etcdrpc]: ../etcdserver/etcdserverpb/rpc.proto
[storagerpc]: ../mvcc/mvccpb/kv.proto
[member_list_rpc]: ../etcdserver/etcdserverpb/rpc.proto#L493-L497
