load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "date.go",
        "error.go",
        "in.go",
        "length.go",
        "match.go",
        "minmax.go",
        "multipleof.go",
        "not_in.go",
        "not_nil.go",
        "required.go",
        "string.go",
        "struct.go",
        "util.go",
        "validation.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/go-ozzo/ozzo-validation",
    importpath = "github.com/go-ozzo/ozzo-validation",
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
        "//vendor/github.com/go-ozzo/ozzo-validation/is:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
