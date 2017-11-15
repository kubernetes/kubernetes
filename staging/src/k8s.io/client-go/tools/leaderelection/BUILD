load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["leaderelection.go"],
    importpath = "k8s.io/client-go/tools/leaderelection",
    visibility = ["//visibility:public"],
    deps = [
        "//tools/leaderelection/resourcelock:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/wait:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["leaderelection_test.go"],
    importpath = "k8s.io/client-go/tools/leaderelection",
    library = ":go_default_library",
    deps = [
        "//kubernetes/typed/core/v1/fake:go_default_library",
        "//testing:go_default_library",
        "//tools/leaderelection/resourcelock:go_default_library",
        "//tools/record:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//staging/src/k8s.io/client-go/tools/leaderelection/resourcelock:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
