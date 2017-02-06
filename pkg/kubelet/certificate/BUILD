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
        "certificate_manager.go",
        "certificate_store.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/apis/certificates/v1beta1:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/certificates/v1beta1:go_default_library",
        "//pkg/util:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/fields",
        "//vendor:k8s.io/apimachinery/pkg/util/wait",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/client-go/util/cert",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "certificate_manager_test.go",
        "certificate_store_test.go",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/apis/certificates/v1beta1:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/certificates/v1beta1:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/client-go/util/cert",
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
