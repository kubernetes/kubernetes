load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "fischer.go",
        "flunder.go",
        "interface.go",
    ],
    importpath = "k8s.io/sample-apiserver/pkg/client/informers/internalversion/wardle/internalversion",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/apis/wardle:go_default_library",
        "//pkg/client/clientset/internalversion:go_default_library",
        "//pkg/client/informers/internalversion/internalinterfaces:go_default_library",
        "//pkg/client/listers/wardle/internalversion:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/watch:go_default_library",
        "//vendor/k8s.io/client-go/tools/cache:go_default_library",
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
    visibility = ["//visibility:public"],
)
