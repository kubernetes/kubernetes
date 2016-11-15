package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_library(
    name = "go_default_library",
    srcs = ["i18n.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/generated:go_default_library",
        "//pkg/util/yaml:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/nicksnyder/go-i18n/i18n/bundle",
        "//vendor:golang.org/x/text/language",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["i18n_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [],
)
