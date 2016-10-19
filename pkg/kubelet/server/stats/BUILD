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
        "doc.go",
        "fs_resource_analyzer.go",
        "handler.go",
        "resource_analyzer.go",
        "summary.go",
        "volume_stat_calculator.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/unversioned:go_default_library",
        "//pkg/kubelet/api/v1alpha1/stats:go_default_library",
        "//pkg/kubelet/cm:go_default_library",
        "//pkg/kubelet/container:go_default_library",
        "//pkg/kubelet/leaky:go_default_library",
        "//pkg/kubelet/network:go_default_library",
        "//pkg/kubelet/types:go_default_library",
        "//pkg/kubelet/util/format:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/util/wait:go_default_library",
        "//pkg/volume:go_default_library",
        "//vendor:github.com/emicklei/go-restful",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/google/cadvisor/info/v1",
        "//vendor:github.com/google/cadvisor/info/v2",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "mocks_test.go",
        "summary_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/unversioned:go_default_library",
        "//pkg/kubelet/api/v1alpha1/stats:go_default_library",
        "//pkg/kubelet/cm:go_default_library",
        "//pkg/kubelet/container:go_default_library",
        "//pkg/kubelet/leaky:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/volume:go_default_library",
        "//vendor:github.com/google/cadvisor/info/v1",
        "//vendor:github.com/google/cadvisor/info/v2",
        "//vendor:github.com/google/gofuzz",
        "//vendor:github.com/stretchr/testify/assert",
        "//vendor:github.com/stretchr/testify/mock",
    ],
)
