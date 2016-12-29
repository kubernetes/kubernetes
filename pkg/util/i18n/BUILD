package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["i18n.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/generated:go_default_library",
        "//vendor:github.com/chai2010/gettext-go/gettext",
        "//vendor:github.com/golang/glog",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["i18n_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
)
