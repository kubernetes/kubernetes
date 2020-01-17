load("@io_bazel_rules_go//go:def.bzl", "go_library")

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
    importmap = "k8s.io/kubernetes/vendor/k8s.io/system-validators/validators",
    importpath = "k8s.io/system-validators/validators",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/blang/semver:go_default_library",
        "//vendor/github.com/pkg/errors:go_default_library",
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
