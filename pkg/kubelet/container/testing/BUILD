package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "fake_cache.go",
        "fake_runtime.go",
        "mockfileinfo.go",
        "os.go",
        "runtime_mock.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1:go_default_library",
        "//pkg/kubelet/container:go_default_library",
        "//pkg/util/flowcontrol:go_default_library",
        "//pkg/util/term:go_default_library",
        "//pkg/volume:go_default_library",
        "//vendor:github.com/golang/mock/gomock",
        "//vendor:github.com/stretchr/testify/mock",
        "//vendor:k8s.io/apimachinery/pkg/types",
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
