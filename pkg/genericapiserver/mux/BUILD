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
        "container.go",
        "doc.go",
        "pathrecorder.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/genericapiserver/api/handlers/responsewriters:go_default_library",
        "//vendor:github.com/emicklei/go-restful",
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["container_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = ["//vendor:github.com/stretchr/testify/assert"],
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
