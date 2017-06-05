package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "authentication_client.go",
        "doc.go",
        "generated_expansion.go",
        "tokenreview.go",
        "tokenreview_expansion.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/apis/authentication:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/scheme:go_default_library",
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
        "//pkg/client/clientset_generated/internalclientset/typed/authentication/internalversion/fake:all-srcs",
    ],
    tags = ["automanaged"],
)
