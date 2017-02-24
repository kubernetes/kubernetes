package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["openapi.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/openapi",
        "//vendor:k8s.io/gengo/args",
        "//vendor:k8s.io/gengo/generator",
        "//vendor:k8s.io/gengo/namer",
        "//vendor:k8s.io/gengo/types",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["openapi_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/stretchr/testify/assert",
        "//vendor:k8s.io/gengo/generator",
        "//vendor:k8s.io/gengo/namer",
        "//vendor:k8s.io/gengo/parser",
        "//vendor:k8s.io/gengo/types",
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
