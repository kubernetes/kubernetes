package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["main.go"],
    deps = [
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apiextensions-apiserver/examples/client-go/apis/cr/v1:go_default_library",
        "//vendor/k8s.io/apiextensions-apiserver/examples/client-go/client:go_default_library",
        "//vendor/k8s.io/apiextensions-apiserver/examples/client-go/controller:go_default_library",
        "//vendor/k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/client-go/rest:go_default_library",
        "//vendor/k8s.io/client-go/tools/clientcmd:go_default_library",
    ],
)

go_binary(
    name = "client-go",
    library = ":go_default_library",
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
        "//staging/src/k8s.io/apiextensions-apiserver/examples/client-go/apis/cr/v1:all-srcs",
        "//staging/src/k8s.io/apiextensions-apiserver/examples/client-go/client:all-srcs",
        "//staging/src/k8s.io/apiextensions-apiserver/examples/client-go/controller:all-srcs",
    ],
    tags = ["automanaged"],
)
