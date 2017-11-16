load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = select({
        "@io_bazel_rules_go//go/platform:darwin_amd64": [
            "fds_other.go",
        ],
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "fds_linux.go",
        ],
        "@io_bazel_rules_go//go/platform:windows_amd64": [
            "fds_other.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/coreos/etcd/pkg/runtime",
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
