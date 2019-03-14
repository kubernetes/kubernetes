load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "idna.go",
        "punycode.go",
        "tables.go",
        "trie.go",
        "trieval.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/net/idna",
    importpath = "golang.org/x/net/idna",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/golang.org/x/text/secure/bidirule:go_default_library",
        "//vendor/golang.org/x/text/unicode/bidi:go_default_library",
        "//vendor/golang.org/x/text/unicode/norm:go_default_library",
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
