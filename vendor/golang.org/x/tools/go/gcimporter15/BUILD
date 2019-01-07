load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "bexport.go",
        "bimport.go",
        "exportdata.go",
        "gcimporter.go",
        "isAlias18.go",
        "isAlias19.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/tools/go/gcimporter15",
    importpath = "golang.org/x/tools/go/gcimporter15",
    visibility = ["//visibility:public"],
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
