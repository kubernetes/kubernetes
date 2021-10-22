# Example

Example of using the AWS SDK for Go with an HTTPS_PROXY that requires client
TLS certificates. The example will use the proxy configured via the environment
variable `HTTPS_PROXY` proxy a request for the Amazon S3 `ListBuckets` API
operation call.

The example assumes credentials are provided in the environment, or shared
credentials file `~/.aws/credentials`. The `certificate` and `key` files paths
are required to be specified when the example is run. An certificate authority
(CA) file path can also be optionally specified.

Refer to [httpproxy.FromEnvironment](https://godoc.org/golang.org/x/net/http/httpproxy#FromEnvironment)
for details using `HTTPS_PROXY` with the Go HTTP client. 

## Usage:

```sh
export HTTPS_PROXY=https://127.0.0.1:8443
export AWS_REGION=us-west-2
go run -cert <certfile> -key <keyfile> [-ca <cafile>]
```

