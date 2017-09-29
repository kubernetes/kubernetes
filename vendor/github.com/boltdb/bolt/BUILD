load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "bolt_unix.go",
        "boltsync_unix.go",
        "bucket.go",
        "cursor.go",
        "db.go",
        "doc.go",
        "errors.go",
        "freelist.go",
        "node.go",
        "page.go",
        "tx.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:darwin_amd64": [
            "bolt_amd64.go",
        ],
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "bolt_amd64.go",
            "bolt_linux.go",
        ],
        "@io_bazel_rules_go//go/platform:windows_amd64": [
            "bolt_amd64.go",
            "bolt_windows.go",
        ],
        "//conditions:default": [],
    }),
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
