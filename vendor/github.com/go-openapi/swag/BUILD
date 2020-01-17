load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "convert.go",
        "convert_types.go",
        "doc.go",
        "json.go",
        "loading.go",
        "name_lexem.go",
        "net.go",
        "path.go",
        "post_go18.go",
        "post_go19.go",
        "pre_go18.go",
        "pre_go19.go",
        "split.go",
        "util.go",
        "yaml.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/go-openapi/swag",
    importpath = "github.com/go-openapi/swag",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/mailru/easyjson/jlexer:go_default_library",
        "//vendor/github.com/mailru/easyjson/jwriter:go_default_library",
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
