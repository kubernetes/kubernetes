load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "cmd.go",
        "const.go",
        "e2erunner.go",
        "env.go",
        "tar.go",
    ],
    importpath = "k8s.io/kubernetes/cluster/images/conformance/go-runner",
    visibility = ["//visibility:private"],
    deps = ["//vendor/github.com/pkg/errors:go_default_library"],
)

go_binary(
    name = "go-runner",
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
    srcs = [
        "cmd_test.go",
        "env_test.go",
        "tar_test.go",
    ],
    embed = [":go_default_library"],
    deps = ["//vendor/github.com/pkg/errors:go_default_library"],
)
