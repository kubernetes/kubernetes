load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "bexport.go",
        "bimport.go",
        "exportdata.go",
        "gcimporter.go",
        "iexport.go",
        "iimport.go",
        "newInterface10.go",
        "newInterface11.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/tools/go/internal/gcimporter",
    importpath = "golang.org/x/tools/go/internal/gcimporter",
    visibility = ["//vendor/golang.org/x/tools/go:__subpackages__"],
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
