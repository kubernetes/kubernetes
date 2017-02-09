package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "fake_authentication_client.go",
        "fake_generated_expansion.go",
        "fake_tokenreview.go",
        "fake_tokenreview_expansion.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/apis/authentication:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/authentication/internalversion:go_default_library",
        "//vendor:k8s.io/client-go/rest",
        "//vendor:k8s.io/client-go/testing",
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
