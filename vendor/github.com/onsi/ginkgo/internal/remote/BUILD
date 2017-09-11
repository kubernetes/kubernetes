load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "aggregator.go",
        "forwarding_reporter.go",
        "output_interceptor.go",
        "server.go",
        "syscall_dup_unix.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:darwin_amd64": [
            "output_interceptor_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "output_interceptor_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:windows_amd64": [
            "output_interceptor_win.go",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//vendor/github.com/onsi/ginkgo:__subpackages__"],
    deps = [
        "//vendor/github.com/onsi/ginkgo/config:go_default_library",
        "//vendor/github.com/onsi/ginkgo/internal/spec_iterator:go_default_library",
        "//vendor/github.com/onsi/ginkgo/reporters:go_default_library",
        "//vendor/github.com/onsi/ginkgo/reporters/stenographer:go_default_library",
        "//vendor/github.com/onsi/ginkgo/types:go_default_library",
    ],
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
