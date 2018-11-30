load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["types.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/mesos/mesos-go/api/v1/lib/encoding",
    importpath = "github.com/mesos/mesos-go/api/v1/lib/encoding",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/gogo/protobuf/proto:go_default_library",
        "//vendor/github.com/mesos/mesos-go/api/v1/lib/encoding/framing:go_default_library",
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
        "//vendor/github.com/mesos/mesos-go/api/v1/lib/encoding/codecs:all-srcs",
        "//vendor/github.com/mesos/mesos-go/api/v1/lib/encoding/framing:all-srcs",
        "//vendor/github.com/mesos/mesos-go/api/v1/lib/encoding/json:all-srcs",
        "//vendor/github.com/mesos/mesos-go/api/v1/lib/encoding/proto:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
