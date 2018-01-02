# CSI Go Validation

This package is used to validate the CSI specification with Go language bindings.

## Build Reference

To validate the Go language bindings against the current specification use the following command:

```bash
$ make
```

The above command will download the `protoc` and `protoc-gen-go` binaries if they are not present and then proceed to build the CSI Go language bindings.

### Environment Variables

The following table lists the environment variables that can be used to influence the behavior of the Makefile:

| Name | Default Value | Description |
|------|---------------|-------------|
| `PROTOC_VER` | `3.3.0` | The version of the protoc binary. |
