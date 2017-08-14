package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["patch_test.go"],
    library = ":go_default_library",
    deps = [
        "//vendor/github.com/davecgh/go-spew/spew:go_default_library",
        "//vendor/github.com/evanphx/json-patch:go_default_library",
        "//vendor/github.com/ghodss/yaml:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/json:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = ["patch.go"],
    deps = [
        "//vendor/github.com/evanphx/json-patch:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/json:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/mergepatch:go_default_library",
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
)
