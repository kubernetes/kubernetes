# HTTP JSON Error Schema

The `error.proto` represents the HTTP-JSON schema used by Google APIs to convey
error payloads as described by https://cloud.google.com/apis/design/errors#http_mapping.
This package is for internal parsing logic only and should not be used in any
other context.

## Regeneration

To regenerate the protobuf Go code you will need the following:

* A local copy of [googleapis], the absolute path to which should be exported to
the environment variable `GOOGLEAPIS`
* The protobuf compiler [protoc]
* The Go [protobuf plugin]
* The [goimports] tool

From this directory run the following command:
```sh
protoc -I $GOOGLEAPIS -I. --go_out=. --go_opt=module=github.com/googleapis/gax-go/v2/apierror/internal/proto error.proto
goimports -w .
```

Note: the `module` plugin option ensures the generated code is placed in this
directory, and not in several nested directories defined by `go_package` option.

[googleapis]: https://github.com/googleapis/googleapis
[protoc]: https://github.com/protocolbuffers/protobuf#protocol-compiler-installation
[protobuf plugin]: https://developers.google.com/protocol-buffers/docs/reference/go-generated
[goimports]: https://pkg.go.dev/golang.org/x/tools/cmd/goimports