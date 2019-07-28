load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "aggregator.go",
        "forwarding_reporter.go",
        "output_interceptor.go",
        "output_interceptor_unix.go",
        "output_interceptor_win.go",
        "server.go",
        "syscall_dup_linux_arm64.go",
        "syscall_dup_solaris.go",
        "syscall_dup_unix.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/onsi/ginkgo/internal/remote",
    importpath = "github.com/onsi/ginkgo/internal/remote",
    visibility = ["//vendor/github.com/onsi/ginkgo:__subpackages__"],
    deps = [
        "//vendor/github.com/onsi/ginkgo/config:go_default_library",
        "//vendor/github.com/onsi/ginkgo/internal/spec_iterator:go_default_library",
        "//vendor/github.com/onsi/ginkgo/internal/writer:go_default_library",
        "//vendor/github.com/onsi/ginkgo/reporters:go_default_library",
        "//vendor/github.com/onsi/ginkgo/reporters/stenographer:go_default_library",
        "//vendor/github.com/onsi/ginkgo/types:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:darwin": [
            "//vendor/github.com/hpcloud/tail:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "//vendor/github.com/hpcloud/tail:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//vendor/github.com/hpcloud/tail:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/github.com/hpcloud/tail:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "//vendor/github.com/hpcloud/tail:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "//vendor/github.com/hpcloud/tail:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "//vendor/github.com/hpcloud/tail:go_default_library",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
