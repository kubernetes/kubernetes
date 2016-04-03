# Hacking

### Godep

We use [go-extpoints](https://github.com/progrium/go-extpoints) to generate
code, which is included via Godeps. When you want to update the dependencies,
make sure to run `godep save ./... github.com/progrium/go-extpoints` instead
of the usual `godep save ./...` to not remove the vendored `go-extpoints`.

Even if you do, Travis should catch that since `hooks/run-extpoints.sh`
depends on those vendored source files.
