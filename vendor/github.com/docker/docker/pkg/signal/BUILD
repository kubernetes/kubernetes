load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "signal.go",
        "signal_darwin.go",
        "signal_freebsd.go",
        "signal_linux.go",
        "signal_unix.go",
        "signal_unsupported.go",
        "signal_windows.go",
        "trap.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/docker/docker/pkg/signal",
    importpath = "github.com/docker/docker/pkg/signal",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/pkg/errors:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux": [
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
        "//vendor/github.com/docker/docker/pkg/signal/testfiles:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
