package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "cgroup_validator.go",
        "docker_validator.go",
        "kernel_validator.go",
        "kernel_validator_helper.go",
        "os_validator.go",
        "package_validator.go",
        "report.go",
        "types.go",
        "validators.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "types_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "types_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "types_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "types_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "types_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "types_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "types_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "types_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "types_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "types_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "types_windows.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "k8s.io/kubernetes/test/e2e_node/system",
    deps = [
        "//vendor/github.com/blang/semver:go_default_library",
        "//vendor/github.com/docker/docker/api/types:go_default_library",
        "//vendor/github.com/docker/docker/client:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/golang.org/x/net/context:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/errors:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "cgroup_validator_test.go",
        "docker_validator_test.go",
        "kernel_validator_test.go",
        "os_validator_test.go",
        "package_validator_test.go",
    ],
    embed = [":go_default_library"],
    tags = ["e2e"],
    deps = [
        "//vendor/github.com/docker/docker/api/types:go_default_library",
        "//vendor/github.com/stretchr/testify/assert:go_default_library",
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
)
