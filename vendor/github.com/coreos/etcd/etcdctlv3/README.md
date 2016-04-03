etcdctl
========

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

- order -- order of results; ASCEND or DESCEND

- sort-by -- sort target; CREATE, KEY, MODIFY, VALUE, or VERSION

TODO: add consistency, from, prefix

#### Return value

##### Simple reply

- \<key\>\n\<value\>\n\<next_key\>\n\<next_value\>...

- Error string if GET failed. Exit code is non-zero.

##### JSON reply

The JSON encoding of the [RPC message][etcdrpc] for a key-value pair for each fetched key-value.

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

TODO: --prefix, --from

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

### WATCH [options] [key or prefix]

Watch watches events stream on keys or prefixes. The watch command runs until it encounters an error or is terminated by the user.

#### Options

- hex -- print out key and value as hex encode string

- interactive -- begins an interactive watch session

- prefix -- watch on a prefix if prefix is set.

- rev -- the revision to start watching. Specifying a revision is useful for observing past events.

#### Input Format

Input is only accepted for interactive mode.

```
watch [options] <key or prefix>\n
```

#### Return value

##### Simple reply

- \<event\>\n\<key\>\n\<value\>\n\<event\>\n\<next_key\>\n\<next_value\>\n...

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

## Utility Commands

### LOCK \<lockname\>

LOCK acquires a distributed named mutex with a given name. Once the lock is acquired, it will be held until etcdctlv3 is terminated.

#### Return value

- Once the lock is acquired, the result for the GET on the unique lock holder key is displayed.

- LOCK returns a zero exit code only if it is terminated by a signal and can release the lock.

#### Example
```bash
./etcdctl lock mylock
mylock/1234534535445


```

### Notes

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

### Notes

The lease length of a leader defaults to 60 seconds. If a candidate is abnormally terminated, election
progress may be delayed by up to 60 seconds.


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


## Notes

- JSON encoding for keys and values uses base64 since they are byte strings.


[etcdrpc]: ../etcdserver/etcdserverpb/rpc.proto
[storagerpc]: ../storage/storagepb/kv.proto

## Compatibility Support

etcdctl is still in its early stage. We try out best to ensure fully compatible releases, however we might break compatibility to fix bugs or improve commands. If we intend to release a version of etcdctl with backward incompatibilities, we will provide notice prior to release and have instructions on how to upgrade.

### Input Compatibility

Input includes the command name, its flags, and its arguments. We ensure backward compatibility of the input of normal commands in non-interactive mode.

### Output Compatibility

Output includes output from etcdctl and its exit code. etcdctl provides `simple` output format by default.
We ensure compatibility for the `simple` output format of normal commands in non-interactive mode. Currently, we do not ensure
backward compatibility for `JSON` format and the format in non-interactive mode. Currently, we do not ensure backward compatibility of utility commands.

### TODO: compatibility with etcd server
