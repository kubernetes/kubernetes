load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "capabilities.go",
        "capabilities_unsupported.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/opencontainers/runc/libcontainer/capabilities",
    importpath = "github.com/opencontainers/runc/libcontainer/capabilities",
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:android": [
            "//vendor/github.com/opencontainers/runc/libcontainer/configs:go_default_library",
            "//vendor/github.com/sirupsen/logrus:go_default_library",
            "//vendor/github.com/syndtr/gocapability/capability:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/github.com/opencontainers/runc/libcontainer/configs:go_default_library",
            "//vendor/github.com/sirupsen/logrus:go_default_library",
            "//vendor/github.com/syndtr/gocapability/capability:go_default_library",
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
