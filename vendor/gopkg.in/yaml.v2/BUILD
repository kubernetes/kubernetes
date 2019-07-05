load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "apic.go",
        "decode.go",
        "emitterc.go",
        "encode.go",
        "parserc.go",
        "readerc.go",
        "resolve.go",
        "scannerc.go",
        "sorter.go",
        "writerc.go",
        "yaml.go",
        "yamlh.go",
        "yamlprivateh.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/gopkg.in/yaml.v2",
    importpath = "gopkg.in/yaml.v2",
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
