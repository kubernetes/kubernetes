
## Why grpc-gateway

etcd v3 uses [gRPC][grpc] for its messaging protocol. The etcd project includes a gRPC-based [Go client][go-client] and a command line utility, [etcdctl][etcdctl], for communicating with an etcd cluster through gRPC. For languages with no gRPC support, etcd provides a JSON [grpc-gateway][grpc-gateway]. This gateway serves a RESTful proxy that translates HTTP/JSON requests into gRPC messages.


## Using grpc-gateway

The gateway accepts a [JSON mapping][json-mapping] for etcd's [protocol buffer][api-ref] message definitions. Note that `key` and `value` fields are defined as byte arrays and therefore must be base64 encoded in JSON.

```bash
<<COMMENT
https://www.base64encode.org/
foo is 'Zm9v' in Base64
bar is 'YmFy'
COMMENT

curl -L http://localhost:2379/v3alpha/kv/put \
	-X POST -d '{"key": "Zm9v", "value": "YmFy"}'

curl -L http://localhost:2379/v3alpha/kv/range \
	-X POST -d '{"key": "Zm9v"}'
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

