# Migrate applications from using API v2 to API v3

The data store v2 is still accessible from the API v2 after upgrading to etcd3. Thus, it will work as before and require no application changes. With etcd 3, applications use the new grpc API v3 to access the mvcc store, which provides more features and improved performance. The mvcc store and the old store v2 are separate and isolated; writes to the store v2 will not affect the mvcc store and, similarly, writes to the mvcc store will not affect the store v2.

Migrating an application from the API v2 to the API v3 involves two steps: 1) migrate the client library and, 2) migrate the data. If the application can rebuild the data, then migrating the data is unnecessary.

## Migrate client library

API v3 is different from API v2, thus application developers need to use a new client library to send requests to etcd API v3. The documentation of the client v3 is available at https://godoc.org/github.com/coreos/etcd/clientv3. 

There are some notable differences between API v2 and API v3:

- Transaction: In v3, etcd provides multi-key conditional transactions. Applications should use transactions in place of `Compare-And-Swap` operations.

- Flat key space: There are no directories in API v3, only keys. For example, "/a/b/c/" is a key. Range queries support getting all keys matching a given prefix.

- Compacted responses: Operations like `Delete` no longer return previous values. To get the deleted value, a transaction can be used to atomically get the key and then delete its value.

- Leases: A replacement for v2 TTLs; the TTL is bound to a lease and keys attach to the lease. When the TTL expires, the lease is revoked and all attached keys are removed.

## Migrate data

Application data can be migrated either offline or online. Offline migration is much simpler than online migration and is recommended.

### Offline migration

Offline migration is very simple but requires etcd downtime. If an etcd downtime window spanning from seconds to minutes is acceptable, offline migration is a good choice and is easy to automate.

First, all members in the etcd cluster must converge to the same state. This can be achieved by stopping all applications that write keys to etcd. Alternatively, if the applications must remain running, configure etcd to listen on a different client URL and restart all etcd members. To check if the states converged, within a few seconds, use the `ETCDCTL_API=3 etcdctl endpoint status` command to confirm that the `raft index` of all members match (or differ by at most 1 due to an internal sync raft command).

Second, migrate the v2 keys into v3 with the [migrate][migrate_command] (`ETCDCTL_API=3 etcdctl migrate`) command. The migrate command writes keys in the v2 store to a user-provided transformer program and reads back transformed keys. It then writes transformed keys into the mvcc store. This usually takes at most tens of seconds.

Restart the etcd members and everything should just work.

### Online migration

If the application cannot tolerate any downtime, then it must migrate online. The implementation of online migration will vary from application to application but the overall idea is the same.

First, write application code using the v3 API. The application must support two modes: a migration mode and a normal mode. The application starts in migration mode. When running in migration mode, the application reads keys using the v3 API first, and, if it cannot find the key, it retries with the API v2. In normal mode, the application only reads keys using the v3 API. The application writes keys over the API v3 in both modes. To acknowledge a switch from migration mode to normal mode, the application watches on a switch mode key. When switch key’s value turns to `true`, the application switches over from migration mode to normal mode.

Second, start a background job to migrate data from the store v2 to the mvcc store by reading keys from the API v2 and writing keys to the API v3. 

After finishing data migration, the background job writes `true` into the switch mode key to notify the application that it may switch modes.

Online migration can be difficult when the application logic depends on store v2 indexes. Applications will need additional logic to convert mvcc store revisions to store v2 indexes.

[migrate_command]: ../../etcdctl/README.md#migrate-options
