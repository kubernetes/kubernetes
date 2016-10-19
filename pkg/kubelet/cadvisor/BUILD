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
        "cadvisor_linux.go",
        "doc.go",
        "types.go",
        "util.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/resource:go_default_library",
        "//pkg/kubelet/types:go_default_library",
        "//pkg/util/runtime:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/google/cadvisor/cache/memory",
        "//vendor:github.com/google/cadvisor/container",
        "//vendor:github.com/google/cadvisor/events",
        "//vendor:github.com/google/cadvisor/fs",
        "//vendor:github.com/google/cadvisor/http",
        "//vendor:github.com/google/cadvisor/info/v1",
        "//vendor:github.com/google/cadvisor/info/v2",
        "//vendor:github.com/google/cadvisor/manager",
        "//vendor:github.com/google/cadvisor/metrics",
        "//vendor:github.com/google/cadvisor/utils/sysfs",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["cadvisor_linux_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/kubelet/types:go_default_library",
        "//vendor:github.com/google/cadvisor/info/v1",
        "//vendor:github.com/google/cadvisor/metrics",
    ],
)
