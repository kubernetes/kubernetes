load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "kind_string.go",
        "tables10.0.0.go",
        "tables11.0.0.go",
        "tables9.0.0.go",
        "transform.go",
        "trieval.go",
        "width.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/text/width",
    importpath = "golang.org/x/text/width",
    visibility = ["//visibility:public"],
    deps = ["//vendor/golang.org/x/text/transform:go_default_library"],
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
