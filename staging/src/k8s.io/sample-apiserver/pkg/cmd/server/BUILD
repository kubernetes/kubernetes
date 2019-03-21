package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["start.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/sample-apiserver/pkg/cmd/server",
    importpath = "k8s.io/sample-apiserver/pkg/cmd/server",
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/errors:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/server:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/server/options:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//staging/src/k8s.io/sample-apiserver/pkg/admission/plugin/banflunder:go_default_library",
        "//staging/src/k8s.io/sample-apiserver/pkg/admission/wardleinitializer:go_default_library",
        "//staging/src/k8s.io/sample-apiserver/pkg/apis/wardle/v1alpha1:go_default_library",
        "//staging/src/k8s.io/sample-apiserver/pkg/apiserver:go_default_library",
        "//staging/src/k8s.io/sample-apiserver/pkg/generated/clientset/versioned:go_default_library",
        "//staging/src/k8s.io/sample-apiserver/pkg/generated/informers/externalversions:go_default_library",
        "//vendor/github.com/spf13/cobra:go_default_library",
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
