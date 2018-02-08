package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "initializer.go",
        "interfaces.go",
    ],
    importpath = "k8s.io/apiserver/pkg/admission/initializer",
    deps = [
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/admission:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/authorization/authorizer:go_default_library",
        "//vendor/k8s.io/client-go/informers:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes:go_default_library",
    ],
)

go_test(
    name = "go_default_xtest",
    srcs = ["initializer_test.go"],
    importpath = "k8s.io/apiserver/pkg/admission/initializer_test",
    deps = [
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/admission:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/admission/initializer:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/authorization/authorizer:go_default_library",
        "//vendor/k8s.io/client-go/informers:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/fake:go_default_library",
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
