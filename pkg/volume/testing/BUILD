package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "mock_volume.go",
        "testing.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/resource:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/client/clientset_generated/clientset:go_default_library",
        "//pkg/cloudprovider:go_default_library",
        "//pkg/util/io:go_default_library",
        "//pkg/util/mount:go_default_library",
        "//pkg/util/strings:go_default_library",
        "//pkg/util/testing:go_default_library",
        "//pkg/util/uuid:go_default_library",
        "//pkg/volume:go_default_library",
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
