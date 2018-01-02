# gRPC naming and discovery

etcd provides a gRPC resolver to support an alternative name system that fetches endpoints from etcd for discovering gRPC services. The underlying mechanism is based on watching updates to keys prefixed with the service name.

## Using etcd discovery with go-grpc

The etcd client provides a gRPC resolver for resolving gRPC endpoints with an etcd backend. The resolver is initialized with an etcd client and given a target for resolution:

```go
import (
	"github.com/coreos/etcd/clientv3"
	etcdnaming "github.com/coreos/etcd/clientv3/naming"

	"google.golang.org/grpc"
)

...

cli, cerr := clientv3.NewFromURL("http://localhost:2379")
r := &etcdnaming.GRPCResolver{Client: cli}
b := grpc.RoundRobin(r)
conn, gerr := grpc.Dial("my-service", grpc.WithBalancer(b))
```

## Managing service endpoints

The etcd resolver treats all keys under the prefix of the resolution target following a "/" (e.g., "my-service/") with JSON-encoded go-grpc `naming.Update` values as potential service endpoints. Endpoints are added to the service by creating new keys and removed from the service by deleting keys.

### Adding an endpoint

New endpoints can be added to the service through `etcdctl`:

```sh
ETCDCTL_API=3 etcdctl put my-service/1.2.3.4 '{"Addr":"1.2.3.4","Metadata":"..."}'
```

The etcd client's `GRPCResolver.Update` method can also register new endpoints with a key matching the `Addr`:

```go
r.Update(context.TODO(), "my-service", naming.Update{Op: naming.Add, Addr: "1.2.3.4", Metadata: "..."})
```

### Deleting an endpoint

Hosts can be deleted from the service through `etcdctl`:

```sh
ETCDCTL_API=3 etcdctl del my-service/1.2.3.4
```

The etcd client's `GRPCResolver.Update` method also supports deleting endpoints:

```go
r.Update(context.TODO(), "my-service", naming.Update{Op: naming.Delete, Addr: "1.2.3.4"})
```

### Registering an endpoint with a lease

Registering an endpoint with a lease ensures that if the host can't maintain a keepalive heartbeat (e.g., its machine fails), it will be removed from the service:

```sh
lease=`ETCDCTL_API=3 etcdctl lease grant 5 | cut -f2 -d' '`
ETCDCTL_API=3 etcdctl put --lease=$lease my-service/1.2.3.4 '{"Addr":"1.2.3.4","Metadata":"..."}'
ETCDCTL_API=3 etcdctl lease keep-alive $lease
```
