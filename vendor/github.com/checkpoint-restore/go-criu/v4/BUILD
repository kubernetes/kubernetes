load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "main.go",
        "notify.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/checkpoint-restore/go-criu/v4",
    importpath = "github.com/checkpoint-restore/go-criu/v4",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/checkpoint-restore/go-criu/v4/rpc:go_default_library",
        "//vendor/github.com/golang/protobuf/proto:go_default_library",
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
        "//vendor/github.com/checkpoint-restore/go-criu/v4/rpc:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
