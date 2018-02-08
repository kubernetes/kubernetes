load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "client.go",
        "controller.go",
        "event.go",
        "namespace.go",
        "pool.go",
        "rule.go",
        "server_version.go",
        "template.go",
        "util.go",
        "validation.go",
        "volume.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "client_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "client_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "client_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "client_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "client_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "client_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "client_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "client_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "client_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "client_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "client_windows.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/storageos/go-api",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/gorilla/websocket:go_default_library",
        "//vendor/github.com/storageos/go-api/types:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:windows": [
            "//vendor/github.com/Microsoft/go-winio:go_default_library",
        ],
        "//conditions:default": [],
    }),
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
        "//vendor/github.com/storageos/go-api/types:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
