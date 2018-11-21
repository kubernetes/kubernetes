load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "agent.pb.go",
        "agent.pb_ffjson.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/mesos/mesos-go/api/v1/lib/agent",
    importpath = "github.com/mesos/mesos-go/api/v1/lib/agent",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/gogo/protobuf/gogoproto:go_default_library",
        "//vendor/github.com/gogo/protobuf/proto:go_default_library",
        "//vendor/github.com/mesos/mesos-go/api/v1/lib:go_default_library",
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
        "//vendor/github.com/mesos/mesos-go/api/v1/lib/agent/calls:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
