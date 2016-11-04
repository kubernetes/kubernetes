package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "cgroup_validator.go",
        "docker_validator.go",
        "kernel_validator.go",
        "os_validator.go",
        "types.go",
        "util.go",
        "validators.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/util/errors:go_default_library",
        "//vendor:github.com/docker/engine-api/client",
        "//vendor:github.com/docker/engine-api/types",
        "//vendor:github.com/golang/glog",
        "//vendor:golang.org/x/net/context",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "cgroup_validator_test.go",
        "docker_validator_test.go",
        "kernel_validator_test.go",
        "os_validator_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/docker/engine-api/types",
        "//vendor:github.com/stretchr/testify/assert",
    ],
)
