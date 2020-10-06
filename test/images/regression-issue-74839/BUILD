load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "main.go",
        "tcp.go",
    ],
    importpath = "k8s.io/kubernetes/test/images/regression-issue-74839",
    visibility = ["//visibility:private"],
)

go_binary(
    name = "regression-issue-74839",
    embed = [":go_default_library"],
    visibility = ["//visibility:public"],
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

go_test(
    name = "go_default_test",
    srcs = ["tcp_test.go"],
    embed = [":go_default_library"],
)
