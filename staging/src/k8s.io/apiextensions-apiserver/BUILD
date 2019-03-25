package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
)

go_binary(
    name = "apiextensions-apiserver",
    embed = [":go_default_library"],
)

go_library(
    name = "go_default_library",
    srcs = ["main.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apiextensions-apiserver",
    importpath = "k8s.io/apiextensions-apiserver",
    deps = [
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/cmd/server:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/server:go_default_library",
        "//staging/src/k8s.io/component-base/logs:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
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
        "//staging/src/k8s.io/apiextensions-apiserver/examples/client-go:all-srcs",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/apis/apiextensions:all-srcs",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/apiserver:all-srcs",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset:all-srcs",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/client/clientset/internalclientset:all-srcs",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/client/informers/externalversions:all-srcs",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/client/informers/internalversion:all-srcs",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/client/listers/apiextensions/internalversion:all-srcs",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/client/listers/apiextensions/v1beta1:all-srcs",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/cmd/server:all-srcs",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/controller/establish:all-srcs",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/controller/finalizer:all-srcs",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/controller/openapi:all-srcs",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/controller/status:all-srcs",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/crdserverscheme:all-srcs",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/features:all-srcs",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/generated/openapi:all-srcs",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/registry/customresource:all-srcs",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/registry/customresourcedefinition:all-srcs",
        "//staging/src/k8s.io/apiextensions-apiserver/test/integration:all-srcs",
    ],
    tags = ["automanaged"],
)
