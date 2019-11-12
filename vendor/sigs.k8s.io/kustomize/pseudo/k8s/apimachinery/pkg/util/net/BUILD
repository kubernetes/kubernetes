load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "http.go",
        "interface.go",
        "port_range.go",
        "port_split.go",
        "util.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/util/net",
    importpath = "sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/util/net",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/golang.org/x/net/http2:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/util/sets:go_default_library",
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
