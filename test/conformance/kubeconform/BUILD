load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "gen.go",
        "kubeconform.go",
        "link.go",
    ],
    importpath = "k8s.io/kubernetes/test/conformance/kubeconform",
    visibility = ["//visibility:private"],
    deps = [
        "//test/conformance/behaviors:go_default_library",
        "//vendor/github.com/go-openapi/analysis:go_default_library",
        "//vendor/github.com/go-openapi/loads:go_default_library",
        "//vendor/github.com/go-openapi/spec:go_default_library",
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

go_binary(
    name = "kubeconform",
    data = ["//api/openapi-spec"],
    embed = [":go_default_library"],
    visibility = ["//visibility:public"],
)
