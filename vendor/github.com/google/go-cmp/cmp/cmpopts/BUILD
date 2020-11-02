load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "equate.go",
        "ignore.go",
        "sort.go",
        "struct_filter.go",
        "xform.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/google/go-cmp/cmp/cmpopts",
    importpath = "github.com/google/go-cmp/cmp/cmpopts",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/google/go-cmp/cmp:go_default_library",
        "//vendor/github.com/google/go-cmp/cmp/internal/function:go_default_library",
        "//vendor/golang.org/x/xerrors:go_default_library",
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
