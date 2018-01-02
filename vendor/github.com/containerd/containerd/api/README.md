This directory contains the GRPC API definitions for containerd.

All defined services and messages have been aggregated into `*.pb.txt`
descriptors files in this directory. Definitions present here are considered
frozen after the release.

At release time, the current `next.pb.txt` file will be moved into place to
freeze the API changes for the minor version. For example, when 1.0.0 is
released, `next.pb.txt` should be moved to `1.0.txt`. Notice that we leave off
the patch number, since the API will be completely locked down for a given
patch series.

We may find that by default, protobuf descriptors are too noisy to lock down
API changes. In that case, we may filter out certain fields in the descriptors,
possibly regenerating for old versions.

This process is similar to the [process used to ensure backwards compatibility
in Go](https://github.com/golang/go/tree/master/api).
