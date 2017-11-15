load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "fake_authentication_client.go",
        "fake_tokenreview.go",
        "fake_tokenreview_expansion.go",
    ],
    importpath = "k8s.io/client-go/kubernetes/typed/authentication/v1/fake",
    visibility = ["//visibility:public"],
    deps = [
        "//kubernetes/typed/authentication/v1:go_default_library",
        "//rest:go_default_library",
        "//testing:go_default_library",
        "//vendor/k8s.io/api/authentication/v1:go_default_library",
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
