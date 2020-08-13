load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "assertion_format.go",
        "assertion_forward.go",
        "assertion_order.go",
        "assertions.go",
        "doc.go",
        "errors.go",
        "forward_assertions.go",
        "http_assertions.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/stretchr/testify/assert",
    importpath = "github.com/stretchr/testify/assert",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/davecgh/go-spew/spew:go_default_library",
        "//vendor/github.com/pmezard/go-difflib/difflib:go_default_library",
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
