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
        "types_unix.go",
        "types_windows.go",
        "validators.go",
    ],
    importpath = "k8s.io/kubernetes/cmd/kubeadm/app/util/system",
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/util/errors:go_default_library",
        "//vendor/github.com/blang/semver:go_default_library",
        "//vendor/github.com/docker/docker/api/types:go_default_library",
        "//vendor/github.com/docker/docker/client:go_default_library",
        "//vendor/github.com/pkg/errors:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
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
        "//vendor/github.com/pkg/errors:go_default_library",
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
