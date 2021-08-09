load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "main.go",
        "notify.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/checkpoint-restore/go-criu/v5",
    importpath = "github.com/checkpoint-restore/go-criu/v5",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/checkpoint-restore/go-criu/v5/rpc:go_default_library",
        "//vendor/google.golang.org/protobuf/proto:go_default_library",
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
        "//vendor/github.com/checkpoint-restore/go-criu/v5/rpc:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
