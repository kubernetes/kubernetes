load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "gcplogging.go",
        "gcplogging_linux.go",
        "gcplogging_others.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/docker/docker/daemon/logger/gcplogs",
    importpath = "github.com/docker/docker/daemon/logger/gcplogs",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/cloud.google.com/go/compute/metadata:go_default_library",
        "//vendor/cloud.google.com/go/logging:go_default_library",
        "//vendor/github.com/docker/docker/daemon/logger:go_default_library",
        "//vendor/github.com/sirupsen/logrus:go_default_library",
        "//vendor/golang.org/x/net/context:go_default_library",
        "//vendor/google.golang.org/genproto/googleapis/api/monitoredres:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/github.com/docker/docker/dockerversion:go_default_library",
            "//vendor/github.com/docker/docker/pkg/homedir:go_default_library",
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
