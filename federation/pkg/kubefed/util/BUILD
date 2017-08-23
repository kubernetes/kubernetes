package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["util.go"],
    deps = [
        "//federation/apis/federation:go_default_library",
        "//federation/apis/federation/v1beta1:go_default_library",
        "//federation/client/clientset_generated/federation_clientset:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/apis/rbac:go_default_library",
        "//pkg/apis/rbac/v1:go_default_library",
        "//pkg/apis/rbac/v1alpha1:go_default_library",
        "//pkg/apis/rbac/v1beta1:go_default_library",
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//pkg/kubectl/cmd:go_default_library",
        "//pkg/kubectl/cmd/util:go_default_library",
        "//vendor/github.com/spf13/cobra:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/net:go_default_library",
        "//vendor/k8s.io/client-go/discovery:go_default_library",
        "//vendor/k8s.io/client-go/rest:go_default_library",
        "//vendor/k8s.io/client-go/tools/clientcmd:go_default_library",
        "//vendor/k8s.io/client-go/tools/clientcmd/api:go_default_library",
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
