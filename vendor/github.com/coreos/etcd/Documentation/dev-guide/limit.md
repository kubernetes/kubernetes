# System limits

## Request size limit

etcd is designed to handle small key value pairs typical for metadata. Larger requests will work, but may increase the latency of other requests. For the time being, etcd guarantees to support RPC requests with up to 1MB of data. In the future, the size limit may be loosened or made it configurable.

## Storage size limit

The default storage size limit is 2GB, configurable with `--quota-backend-bytes` flag; supports up to 8GB.

