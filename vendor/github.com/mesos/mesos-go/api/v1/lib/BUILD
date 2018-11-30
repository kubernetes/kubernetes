load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "client.go",
        "doc.go",
        "filters.go",
        "fixedpoint.go",
        "labels.go",
        "mesos.pb.go",
        "mesos.pb_ffjson.go",
        "ranges.go",
        "resources.go",
        "values.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/mesos/mesos-go/api/v1/lib",
    importpath = "github.com/mesos/mesos-go/api/v1/lib",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/gogo/protobuf/gogoproto:go_default_library",
        "//vendor/github.com/gogo/protobuf/proto:go_default_library",
        "//vendor/github.com/mesos/mesos-go/api/v1/lib/encoding:go_default_library",
        "//vendor/github.com/mesos/mesos-go/api/v1/lib/roles:go_default_library",
        "//vendor/github.com/pquerna/ffjson/fflib/v1:go_default_library",
    ],
)

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "all-srcs",
    srcs = [
        ":package-srcs",
        "//vendor/github.com/mesos/mesos-go/api/v1/lib/agent:all-srcs",
        "//vendor/github.com/mesos/mesos-go/api/v1/lib/client:all-srcs",
        "//vendor/github.com/mesos/mesos-go/api/v1/lib/debug:all-srcs",
        "//vendor/github.com/mesos/mesos-go/api/v1/lib/encoding:all-srcs",
        "//vendor/github.com/mesos/mesos-go/api/v1/lib/httpcli:all-srcs",
        "//vendor/github.com/mesos/mesos-go/api/v1/lib/recordio:all-srcs",
        "//vendor/github.com/mesos/mesos-go/api/v1/lib/roles:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
