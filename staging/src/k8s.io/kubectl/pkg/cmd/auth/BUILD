load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "auth.go",
        "cani.go",
        "reconcile.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/kubectl/pkg/cmd/auth",
    importpath = "k8s.io/kubectl/pkg/cmd/auth",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/api/authorization/v1:go_default_library",
        "//staging/src/k8s.io/api/rbac/v1:go_default_library",
        "//staging/src/k8s.io/api/rbac/v1alpha1:go_default_library",
        "//staging/src/k8s.io/api/rbac/v1beta1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/meta:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//staging/src/k8s.io/cli-runtime/pkg/genericclioptions:go_default_library",
        "//staging/src/k8s.io/cli-runtime/pkg/printers:go_default_library",
        "//staging/src/k8s.io/cli-runtime/pkg/resource:go_default_library",
        "//staging/src/k8s.io/client-go/discovery:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes/typed/authorization/v1:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes/typed/core/v1:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes/typed/rbac/v1:go_default_library",
        "//staging/src/k8s.io/component-helpers/auth/rbac/reconciliation:go_default_library",
        "//staging/src/k8s.io/kubectl/pkg/cmd/util:go_default_library",
        "//staging/src/k8s.io/kubectl/pkg/describe:go_default_library",
        "//staging/src/k8s.io/kubectl/pkg/scheme:go_default_library",
        "//staging/src/k8s.io/kubectl/pkg/util/rbac:go_default_library",
        "//staging/src/k8s.io/kubectl/pkg/util/templates:go_default_library",
        "//vendor/github.com/spf13/cobra:go_default_library",
        "//vendor/k8s.io/klog/v2:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["cani_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/api/authorization/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/cli-runtime/pkg/genericclioptions:go_default_library",
        "//staging/src/k8s.io/client-go/rest:go_default_library",
        "//staging/src/k8s.io/client-go/rest/fake:go_default_library",
        "//staging/src/k8s.io/kubectl/pkg/cmd/testing:go_default_library",
        "//staging/src/k8s.io/kubectl/pkg/scheme:go_default_library",
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
