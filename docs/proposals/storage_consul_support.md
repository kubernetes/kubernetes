# Storage Interface Simplification To Support Consul as a KVS
----------

## Abstract
-----------

A proposal for a new simplified storage interface with object serialization separated into a higher layer - Raw Storage - in order to reduce the dependencies and requirements for storage backend implementations and to reduce the package dependencies and bootstrap requirements for alternative consumers. The currently existing interface is then implemented by an object serialization layer that wraps around this Raw Storage layer.

## Use Cases
-----------

1. Allow for alternate KV storage backends to be implemented with less replicated code and complexity.  In our case, Consul support.
2. Allow consumers like mesos master election, as defined in contrib/mesos/pkg/election, to make use of this Raw Storage to abstract their KV storage backend without adding complexity to bootstrapping or adding excessive package dependencies.

## Motivation
----------

Many datacenter operators have expressed interest in unifying the KV storage services running on their cluster. (See [#1957](https://github.com/kubernetes/kubernetes/issues/1957)). Usage of EtcD in the code base is not always well-encapsulated, so in order to migrate all EtcD dependencies to an alternate storage solution, some fringe uses of etcd needed to be able to access KV storage without object serialization functionality.

## Raw Storage Implementation
----------

The RawLayer operates on raw data objects independent of the object serialization layer. These raw objects are little more than a byte array and some consensus flavoring sprinkled on them.

```golang
type RawObject struct {
        Data    []byte
        Version uint64
        TTL     int64
}
```

----------
Watches and the events they generate also deal exclusively in raw data objects.

```golang
type RawEvent struct {
        Type        watch.EventType
        Current     RawObject
        Previous    RawObject
        ErrorStatus interface{}
}
type RawWatch interface {
        Stop()
        ResultChan() <-chan RawEvent
}
```
----------
The Raw Storage interface is modeled closely after the existing interface stripped of object serialization requirements and of methods that differ only semantically but not functionally.

```golang
type RawStorage interface {
  Create(ctx context.Context, key string, data []byte, raw *RawObject, ttl uint64) error

  Delete(ctx context.Context, key string, raw *RawObject, preconditions RawFilterFunc) error

  Watch(ctx context.Context, key string, resourceVersion string) (RawWatch, error)

  WatchList(ctx context.Context, key string, resourceVersion string) (RawWatch, error)

  // raw is passed out-by-pointer to prevent additional allocations
  Get(ctx context.Context, key string, raw *RawObject) error

  // In order to prevent multiple deserialization, filtering is performed
  // at the serialization layer
  List(ctx context.Context, key string, rawList *[]RawObject) (uint64, error)

  // raw is an in/out parameter. Set only succeeds if the stored object's
  // Version matches the Version field of the input object. Upon success,
  // the Version field is updated to match the stored object
  Set(ctx context.Context, key string, raw *RawObject) (bool, error)
}
```
[Working implementation](https://github.com/MustWin/kubernetes/blob/consul-integration/pkg/storage/generic/interfaces.go#L34)

## Consul Implementation

We have a reference implementation that is nearing completion that improves test coverage on the storage and watch interfaces and utilizes these strategies. We've already implemented these interfaces for etcd2 ([raw](https://github.com/MustWin/kubernetes/blob/consul-integration/pkg/storage/etcd/etcd_raw.go), [raw watcher](https://github.com/MustWin/kubernetes/blob/consul-integration/pkg/storage/etcd/etcd_raw_watcher.go)), etcd3 ([raw](https://github.com/MustWin/kubernetes/blob/consul-integration/pkg/storage/etcd3/raw.go), [raw watcher](https://github.com/MustWin/kubernetes/blob/consul-integration/pkg/storage/etcd3/raw_watcher.go)), and consul ([raw](https://github.com/MustWin/kubernetes/blob/consul-integration/pkg/storage/consul/consul.go), [raw watcher](https://github.com/MustWin/kubernetes/blob/consul-integration/pkg/storage/consul/consul_watcher.go)).

These raw interfaces are then transformed into generic storage.Interface with a `NewGenericWrapper` [function](https://github.com/MustWin/kubernetes/blob/consul-integration/pkg/storage/generic_implementation.go#L40).

## Contributors

elg0nz
johanatan
mikejihbe
