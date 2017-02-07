package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["start.go"],
    tags = ["automanaged"],
    deps = [
        "//cmd/kube-aggregator/pkg/apis/apiregistration/v1alpha1:go_default_library",
        "//cmd/kube-aggregator/pkg/apiserver:go_default_library",
        "//cmd/kube-aggregator/pkg/legacy:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/client/clientset_generated/clientset:go_default_library",
        "//pkg/kubectl/cmd/util:go_default_library",
        "//vendor:github.com/pborman/uuid",
        "//vendor:github.com/spf13/cobra",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apimachinery/pkg/util/wait",
        "//vendor:k8s.io/apiserver/pkg/registry/generic",
        "//vendor:k8s.io/apiserver/pkg/registry/generic/registry",
        "//vendor:k8s.io/apiserver/pkg/server",
        "//vendor:k8s.io/apiserver/pkg/server/filters",
        "//vendor:k8s.io/apiserver/pkg/server/options",
        "//vendor:k8s.io/apiserver/pkg/storage/storagebackend",
        "//vendor:k8s.io/client-go/rest",
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
