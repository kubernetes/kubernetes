load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "netutil.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "isolate_stub.go",
            "routes.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "isolate_stub.go",
            "routes.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "isolate_stub.go",
            "routes.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "isolate_stub.go",
            "routes.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "isolate_linux.go",
            "routes_linux.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "isolate_stub.go",
            "routes.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "isolate_stub.go",
            "routes.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "isolate_stub.go",
            "routes.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "isolate_stub.go",
            "routes.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "isolate_stub.go",
            "routes.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "isolate_stub.go",
            "routes.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/coreos/etcd/pkg/netutil",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/coreos/etcd/pkg/types:go_default_library",
        "//vendor/github.com/coreos/pkg/capnslog:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/github.com/coreos/etcd/pkg/cpuutil:go_default_library",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
