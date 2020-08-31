load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "display.go",
        "models.go",
        "operations.go",
        "reader.go",
        "writer.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/googleapis/gnostic/jsonschema",
    importpath = "github.com/googleapis/gnostic/jsonschema",
    visibility = ["//visibility:public"],
    deps = ["//vendor/gopkg.in/yaml.v2:go_default_library"],
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
