load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["ebpf_linux.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/opencontainers/runc/libcontainer/cgroups/ebpf",
    importpath = "github.com/opencontainers/runc/libcontainer/cgroups/ebpf",
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:android": [
            "//vendor/github.com/cilium/ebpf:go_default_library",
            "//vendor/github.com/cilium/ebpf/asm:go_default_library",
            "//vendor/github.com/cilium/ebpf/link:go_default_library",
            "//vendor/github.com/pkg/errors:go_default_library",
            "//vendor/github.com/sirupsen/logrus:go_default_library",
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/github.com/cilium/ebpf:go_default_library",
            "//vendor/github.com/cilium/ebpf/asm:go_default_library",
            "//vendor/github.com/cilium/ebpf/link:go_default_library",
            "//vendor/github.com/pkg/errors:go_default_library",
            "//vendor/github.com/sirupsen/logrus:go_default_library",
            "//vendor/golang.org/x/sys/unix:go_default_library",
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
        "//vendor/github.com/opencontainers/runc/libcontainer/cgroups/ebpf/devicefilter:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
