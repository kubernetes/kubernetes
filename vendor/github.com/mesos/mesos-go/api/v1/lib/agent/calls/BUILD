load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "calls.go",
        "calls_generated.go",
        "gen.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/mesos/mesos-go/api/v1/lib/agent/calls",
    importpath = "github.com/mesos/mesos-go/api/v1/lib/agent/calls",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/mesos/mesos-go/api/v1/lib:go_default_library",
        "//vendor/github.com/mesos/mesos-go/api/v1/lib/agent:go_default_library",
        "//vendor/github.com/mesos/mesos-go/api/v1/lib/encoding:go_default_library",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
