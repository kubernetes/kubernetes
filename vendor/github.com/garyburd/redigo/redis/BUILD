load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "conn.go",
        "doc.go",
        "log.go",
        "pool.go",
        "pubsub.go",
        "redis.go",
        "reply.go",
        "scan.go",
        "script.go",
    ],
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/garyburd/redigo/internal:go_default_library"],
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
