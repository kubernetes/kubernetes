### etcd-dump-db

etcd-dump-db inspects etcd db files.

```
Usage:
  etcd-dump-db [command]

Available Commands:
  list-bucket    bucket lists all buckets.
  iterate-bucket iterate-bucket lists key-value pairs in reverse order.
  hash           hash computes the hash of db file.

Flags:
  -h, --help[=false]: help for etcd-dump-db

Use "etcd-dump-db [command] --help" for more information about a command.
```


#### list-bucket [data dir or db file path]

Lists all buckets.

```
$ etcd-dump-db list-bucket agent01/agent.etcd

alarm
auth
authRoles
authUsers
cluster
key
lease
members
members_removed
meta
```


#### hash [data dir or db file path]

Computes the hash of db file.

```
$ etcd-dump-db hash agent01/agent.etcd
db path: agent01/agent.etcd/member/snap/db
Hash: 3700260467


$ etcd-dump-db hash agent02/agent.etcd

db path: agent02/agent.etcd/member/snap/db
Hash: 3700260467


$ etcd-dump-db hash agent03/agent.etcd

db path: agent03/agent.etcd/member/snap/db
Hash: 3700260467
```


#### iterate-bucket [data dir or db file path]

Lists key-value pairs in reverse order.

```
$ etcd-dump-db iterate-bucket agent03/agent.etcd --bucket=key --limit 3

key="\x00\x00\x00\x00\x005@x_\x00\x00\x00\x00\x00\x00\x00\tt", value="\n\x153640412599896088633_9"
key="\x00\x00\x00\x00\x005@x_\x00\x00\x00\x00\x00\x00\x00\bt", value="\n\x153640412599896088633_8"
key="\x00\x00\x00\x00\x005@x_\x00\x00\x00\x00\x00\x00\x00\at", value="\n\x153640412599896088633_7"
```
