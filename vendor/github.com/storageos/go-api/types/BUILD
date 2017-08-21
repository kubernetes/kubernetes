load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "auth.go",
        "capacity_stats.go",
        "controller.go",
        "controller_update_options.go",
        "delete_options.go",
        "deployment.go",
        "driver_instance.go",
        "error_response.go",
        "events.go",
        "list_options.go",
        "namespace.go",
        "operator.go",
        "pool.go",
        "pool_create_options.go",
        "rule.go",
        "template.go",
        "template_create_options.go",
        "version.go",
        "volume.go",
        "volume_create_options.go",
        "volume_update_options.go",
    ],
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
