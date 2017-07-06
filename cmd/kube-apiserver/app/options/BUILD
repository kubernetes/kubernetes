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
        "options.go",
        "validation.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/validation:go_default_library",
        "//pkg/features:go_default_library",
        "//pkg/kubeapiserver/options:go_default_library",
        "//pkg/kubelet/client:go_default_library",
        "//pkg/master/ports:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/net:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/server/options:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/storage/storagebackend:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["options_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = ["//vendor/github.com/spf13/pflag:go_default_library"],
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
