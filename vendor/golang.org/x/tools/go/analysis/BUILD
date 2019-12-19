load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "analysis.go",
        "diagnostic.go",
        "doc.go",
        "validate.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/tools/go/analysis",
    importpath = "golang.org/x/tools/go/analysis",
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
    srcs = [
        ":package-srcs",
        "//vendor/golang.org/x/tools/go/analysis/passes/inspect:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
