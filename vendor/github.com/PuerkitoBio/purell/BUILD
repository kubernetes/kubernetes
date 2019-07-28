load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["purell.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/PuerkitoBio/purell",
    importpath = "github.com/PuerkitoBio/purell",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/PuerkitoBio/urlesc:go_default_library",
        "//vendor/golang.org/x/net/idna:go_default_library",
        "//vendor/golang.org/x/text/unicode/norm:go_default_library",
        "//vendor/golang.org/x/text/width:go_default_library",
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
