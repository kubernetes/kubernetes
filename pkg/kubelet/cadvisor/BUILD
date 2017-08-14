package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "cadvisor_unsupported.go",
        "doc.go",
        "types.go",
        "util.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "cadvisor_linux.go",
        ],
        "@io_bazel_rules_go//go/platform:windows_amd64": [
            "cadvisor_windows.go",
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//vendor/github.com/google/cadvisor/events:go_default_library",
        "//vendor/github.com/google/cadvisor/info/v1:go_default_library",
        "//vendor/github.com/google/cadvisor/info/v2:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "//pkg/kubelet/types:go_default_library",
            "//vendor/github.com/golang/glog:go_default_library",
            "//vendor/github.com/google/cadvisor/cache/memory:go_default_library",
            "//vendor/github.com/google/cadvisor/container:go_default_library",
            "//vendor/github.com/google/cadvisor/fs:go_default_library",
            "//vendor/github.com/google/cadvisor/http:go_default_library",
            "//vendor/github.com/google/cadvisor/manager:go_default_library",
            "//vendor/github.com/google/cadvisor/metrics:go_default_library",
            "//vendor/github.com/google/cadvisor/utils/sysfs:go_default_library",
            "//vendor/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        ],
        "//conditions:default": [],
    }),
)

go_test(
    name = "go_default_test",
    srcs = select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "cadvisor_linux_test.go",
        ],
        "//conditions:default": [],
    }),
    library = ":go_default_library",
    deps = select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "//pkg/kubelet/types:go_default_library",
            "//vendor/github.com/google/cadvisor/info/v1:go_default_library",
            "//vendor/github.com/google/cadvisor/metrics:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//pkg/kubelet/cadvisor/testing:all-srcs",
    ],
    tags = ["automanaged"],
)
