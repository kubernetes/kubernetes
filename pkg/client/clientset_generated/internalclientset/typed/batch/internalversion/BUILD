package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "batch_client.go",
        "cronjob.go",
        "doc.go",
        "generated_expansion.go",
        "job.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/apis/batch:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/scheme:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/watch",
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
    srcs = [
        ":package-srcs",
        "//pkg/client/clientset_generated/internalclientset/typed/batch/internalversion/fake:all-srcs",
    ],
    tags = ["automanaged"],
)
