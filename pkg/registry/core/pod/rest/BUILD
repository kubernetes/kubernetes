package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "log.go",
        "subresources.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/registry/core/pod/rest",
    deps = [
        "//pkg/apis/core:go_default_library",
        "//pkg/apis/core/install:go_default_library",
        "//pkg/apis/core/validation:go_default_library",
        "//pkg/capabilities:go_default_library",
        "//pkg/features:go_default_library",
        "//pkg/kubelet/client:go_default_library",
        "//pkg/registry/core/pod:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/net:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/proxy:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/features:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/registry/generic/registry:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/registry/generic/rest:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/registry/rest:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["log_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//pkg/apis/core:go_default_library",
        "//pkg/registry/registrytest:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/endpoints/request:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/registry/generic:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/registry/generic/registry:go_default_library",
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
