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
        "helpers.go",
        "validate.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/features:go_default_library",
        "//vendor:k8s.io/apis/pkg/api/v1",
        "//vendor:k8s.io/apis/pkg/util",
        "//vendor:k8s.io/apiserver/pkg/util/feature",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["validate_test.go"],
    data = [
        "testdata/profiles",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/stretchr/testify/assert",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apis/pkg/api/v1",
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
