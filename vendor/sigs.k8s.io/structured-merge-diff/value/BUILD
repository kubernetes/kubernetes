load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "fastjson.go",
        "unstructured.go",
        "value.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/structured-merge-diff/value",
    importpath = "sigs.k8s.io/structured-merge-diff/value",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/json-iterator/go:go_default_library",
        "//vendor/gopkg.in/yaml.v2:go_default_library",
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

go_test(
    name = "go_default_test",
    srcs = [
        "less_test.go",
        "unstructured_test.go",
    ],
    embed = [":go_default_library"],
    tags = ["manual"],
    deps = ["//vendor/gopkg.in/yaml.v2:go_default_library"],
)
