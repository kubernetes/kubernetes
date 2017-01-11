package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
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
        "//pkg/api/resource:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/kubelet/types:go_default_library",
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
        "//vendor:k8s.io/apimachinery/pkg/util/runtime",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["cadvisor_linux_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/kubelet/types:go_default_library",
        "//vendor:github.com/google/cadvisor/info/v1",
        "//vendor:github.com/google/cadvisor/metrics",
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
    srcs = [
        ":package-srcs",
        "//pkg/kubelet/cadvisor/testing:all-srcs",
    ],
    tags = ["automanaged"],
)
