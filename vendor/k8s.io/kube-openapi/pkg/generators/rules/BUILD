load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "idl_tag.go",
        "names_match.go",
        "omitempty_match_case.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/kube-openapi/pkg/generators/rules",
    importpath = "k8s.io/kube-openapi/pkg/generators/rules",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/k8s.io/gengo/types:go_default_library",
        "//vendor/k8s.io/kube-openapi/pkg/util/sets:go_default_library",
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
