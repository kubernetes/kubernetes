load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "aggregator.go",
        "forwarding_reporter.go",
        "output_interceptor.go",
        "server.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:darwin": [
            "output_interceptor_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "output_interceptor_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "output_interceptor_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "output_interceptor_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "output_interceptor_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "output_interceptor_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "output_interceptor_unix.go",
            "syscall_dup_solaris.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "output_interceptor_win.go",
        ],
        "//conditions:default": [],
    }) + select({
        "@io_bazel_rules_go//go/platform:android_386": [
            "syscall_dup_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:android_amd64": [
            "syscall_dup_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:android_arm": [
            "syscall_dup_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:android_arm64": [
            "syscall_dup_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin_386": [
            "syscall_dup_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin_amd64": [
            "syscall_dup_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin_arm": [
            "syscall_dup_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin_arm64": [
            "syscall_dup_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly_amd64": [
            "syscall_dup_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd_386": [
            "syscall_dup_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd_amd64": [
            "syscall_dup_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd_arm": [
            "syscall_dup_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:linux_386": [
            "syscall_dup_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "syscall_dup_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:linux_arm": [
            "syscall_dup_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:linux_arm64": [
            "syscall_dup_linux_arm64.go",
        ],
        "@io_bazel_rules_go//go/platform:linux_mips": [
            "syscall_dup_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:linux_mips64": [
            "syscall_dup_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:linux_mips64le": [
            "syscall_dup_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:linux_mipsle": [
            "syscall_dup_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:linux_ppc64": [
            "syscall_dup_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:linux_ppc64le": [
            "syscall_dup_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:linux_s390x": [
            "syscall_dup_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl_386": [
            "syscall_dup_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl_amd64p32": [
            "syscall_dup_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl_arm": [
            "syscall_dup_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd_386": [
            "syscall_dup_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd_amd64": [
            "syscall_dup_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd_arm": [
            "syscall_dup_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd_386": [
            "syscall_dup_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd_amd64": [
            "syscall_dup_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd_arm": [
            "syscall_dup_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9_386": [
            "syscall_dup_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9_amd64": [
            "syscall_dup_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9_arm": [
            "syscall_dup_unix.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/onsi/ginkgo/internal/remote",
    visibility = ["//vendor/github.com/onsi/ginkgo:__subpackages__"],
    deps = [
        "//vendor/github.com/onsi/ginkgo/config:go_default_library",
        "//vendor/github.com/onsi/ginkgo/internal/spec_iterator:go_default_library",
        "//vendor/github.com/onsi/ginkgo/reporters:go_default_library",
        "//vendor/github.com/onsi/ginkgo/reporters/stenographer:go_default_library",
        "//vendor/github.com/onsi/ginkgo/types:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:solaris": [
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
