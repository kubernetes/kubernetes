load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "auth.go",
        "capacity_stats.go",
        "connectivity.go",
        "delete_options.go",
        "deployment.go",
        "driver_instance.go",
        "error_response.go",
        "events.go",
        "health.go",
        "list_options.go",
        "logger.go",
        "namespace.go",
        "node.go",
        "node_update_options.go",
        "operator.go",
        "policy.go",
        "pool.go",
        "pool_options.go",
        "rule.go",
        "template.go",
        "template_create_options.go",
        "user.go",
        "version.go",
        "volume.go",
        "volume_create_options.go",
        "volume_stats.go",
        "volume_update_options.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/storageos/go-api/types",
    importpath = "github.com/storageos/go-api/types",
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
