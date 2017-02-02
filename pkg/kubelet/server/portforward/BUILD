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
        "constants.go",
        "httpstream.go",
        "portforward.go",
        "websocket.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/httpstream",
        "//vendor:k8s.io/apimachinery/pkg/util/httpstream/spdy",
        "//vendor:k8s.io/apimachinery/pkg/util/runtime",
        "//vendor:k8s.io/apiserver/pkg/server/httplog",
        "//vendor:k8s.io/apiserver/pkg/util/wsstream",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "httpstream_test.go",
        "websocket_test.go",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/util/httpstream",
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
