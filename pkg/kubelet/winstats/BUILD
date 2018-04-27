load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:windows": [
            "perfcounter_nodestats.go",
            "perfcounters.go",
            "version.go",
            "winstats.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "k8s.io/kubernetes/pkg/kubelet/winstats",
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:windows": [
            "//vendor/github.com/JeffAshton/win_pdh:go_default_library",
            "//vendor/github.com/golang/glog:go_default_library",
            "//vendor/github.com/google/cadvisor/info/v1:go_default_library",
            "//vendor/github.com/google/cadvisor/info/v2:go_default_library",
            "//vendor/golang.org/x/sys/windows:go_default_library",
            "//vendor/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        ],
        "//conditions:default": [],
    }),
)

go_test(
    name = "go_default_test",
    srcs = select({
        "@io_bazel_rules_go//go/platform:windows": [
            "winstats_test.go",
        ],
        "//conditions:default": [],
    }),
    embed = [":go_default_library"],
    deps = select({
        "@io_bazel_rules_go//go/platform:windows": [
            "//vendor/github.com/google/cadvisor/info/v1:go_default_library",
            "//vendor/github.com/google/cadvisor/info/v2:go_default_library",
            "//vendor/github.com/stretchr/testify/assert:go_default_library",
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
