load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["naming_controller.go"],
    importpath = "k8s.io/apiextensions-apiserver/pkg/controller/status",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/apis/apiextensions:go_default_library",
        "//pkg/client/clientset/internalclientset/typed/apiextensions/internalversion:go_default_library",
        "//pkg/client/informers/internalversion/apiextensions/internalversion:go_default_library",
        "//pkg/client/listers/apiextensions/internalversion:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//vendor/k8s.io/client-go/tools/cache:go_default_library",
        "//vendor/k8s.io/client-go/util/workqueue:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["naming_controller_test.go"],
    importpath = "k8s.io/apiextensions-apiserver/pkg/controller/status",
    library = ":go_default_library",
    deps = [
        "//pkg/apis/apiextensions:go_default_library",
        "//pkg/client/listers/apiextensions/internalversion:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
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
