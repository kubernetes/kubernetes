load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "vmcompute.go",
        "zsyscall_windows.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/Microsoft/hcsshim/internal/vmcompute",
    importpath = "github.com/Microsoft/hcsshim/internal/vmcompute",
    visibility = ["//vendor/github.com/Microsoft/hcsshim:__subpackages__"],
    deps = [
        "//vendor/github.com/Microsoft/hcsshim/internal/interop:go_default_library",
        "//vendor/github.com/Microsoft/hcsshim/internal/log:go_default_library",
        "//vendor/github.com/Microsoft/hcsshim/internal/logfields:go_default_library",
        "//vendor/github.com/Microsoft/hcsshim/internal/oc:go_default_library",
        "//vendor/github.com/Microsoft/hcsshim/internal/timeout:go_default_library",
        "//vendor/go.opencensus.io/trace:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:windows": [
            "//vendor/golang.org/x/sys/windows:go_default_library",
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
