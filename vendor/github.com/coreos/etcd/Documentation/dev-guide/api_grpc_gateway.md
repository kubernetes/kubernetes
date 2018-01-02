
## Why grpc-gateway

etcd v3 uses [gRPC][grpc] for its messaging protocol. The etcd project includes a gRPC-based [Go client][go-client] and a command line utility, [etcdctl][etcdctl], for communicating with an etcd cluster through gRPC. For languages with no gRPC support, etcd provides a JSON [grpc-gateway][grpc-gateway]. This gateway serves a RESTful proxy that translates HTTP/JSON requests into gRPC messages.


## Using grpc-gateway

The gateway accepts a [JSON mapping][json-mapping] for etcd's [protocol buffer][api-ref] message definitions. Note that `key` and `value` fields are defined as byte arrays and therefore must be base64 encoded in JSON.

Use `curl` to put and get a key:

```bash
<<COMMENT
https://www.base64encode.org/
foo is 'Zm9v' in Base64
bar is 'YmFy'
COMMENT

curl -L http://localhost:2379/v3alpha/kv/put \
	-X POST -d '{"key": "Zm9v", "value": "YmFy"}'
# {"header":{"cluster_id":"12585971608760269493","member_id":"13847567121247652255","revision":"2","raft_term":"3"}}

curl -L http://localhost:2379/v3alpha/kv/range \
	-X POST -d '{"key": "Zm9v"}'
# {"header":{"cluster_id":"12585971608760269493","member_id":"13847567121247652255","revision":"2","raft_term":"3"},"kvs":[{"key":"Zm9v","create_revision":"2","mod_revision":"2","version":"1","value":"YmFy"}],"count":"1"}
```

Use `curl` to watch a key:

```bash
curl http://localhost:2379/v3alpha/watch \
        -X POST -d '{"create_request": {"key":"Zm9v"} }' &
# {"result":{"header":{"cluster_id":"12585971608760269493","member_id":"13847567121247652255","revision":"1","raft_term":"2"},"created":true}}

curl -L http://localhost:2379/v3alpha/kv/put \
	-X POST -d '{"key": "Zm9v", "value": "YmFy"}' >/dev/null 2>&1
# {"result":{"header":{"cluster_id":"12585971608760269493","member_id":"13847567121247652255","revision":"2","raft_term":"2"},"events":[{"kv":{"key":"Zm9v","create_revision":"2","mod_revision":"2","version":"1","value":"YmFy"}}]}}
```

## Swagger

Generated [Swagger][swagger] API definitions can be found at [rpc.swagger.json][swagger-doc].

[api-ref]: ./api_reference_v3.md
[go-client]: https://github.com/coreos/etcd/tree/master/clientv3
[etcdctl]: https://github.com/coreos/etcd/tree/master/etcdctl
[grpc]: http://www.grpc.io/
[grpc-gateway]: https://github.com/grpc-ecosystem/grpc-gateway
[json-mapping]: https://developers.google.com/protocol-buffers/docs/proto3#json
[swagger]: http://swagger.io/
[swagger-doc]: apispec/swagger/rpc.swagger.json

