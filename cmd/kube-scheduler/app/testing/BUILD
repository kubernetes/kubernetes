load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["testserver.go"],
    importpath = "k8s.io/kubernetes/cmd/kube-scheduler/app/testing",
    visibility = ["//visibility:public"],
    deps = [
        "//cmd/kube-scheduler/app:go_default_library",
        "//cmd/kube-scheduler/app/config:go_default_library",
        "//cmd/kube-scheduler/app/options:go_default_library",
        "//pkg/scheduler/algorithmprovider/defaults:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//staging/src/k8s.io/client-go/rest:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
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
