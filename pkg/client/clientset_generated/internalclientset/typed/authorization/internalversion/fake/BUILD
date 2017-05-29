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
        "fake_authorization_client.go",
        "fake_generated_expansion.go",
        "fake_localsubjectaccessreview.go",
        "fake_localsubjectaccessreview_expansion.go",
        "fake_selfsubjectaccessreview.go",
        "fake_selfsubjectaccessreview_expansion.go",
        "fake_subjectaccessreview.go",
        "fake_subjectaccessreview_expansion.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/apis/authorization:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/authorization/internalversion:go_default_library",
        "//vendor/k8s.io/client-go/rest:go_default_library",
        "//vendor/k8s.io/client-go/testing:go_default_library",
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
