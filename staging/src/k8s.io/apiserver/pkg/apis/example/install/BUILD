package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["install.go"],
    importpath = "k8s.io/apiserver/pkg/apis/example/install",
    deps = [
        "//vendor/k8s.io/apimachinery/pkg/apimachinery/announced:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apimachinery/registered:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/apis/example:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/apis/example/v1:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["roundtrip_test.go"],
    embed = [":go_default_library"],
    importpath = "k8s.io/apiserver/pkg/apis/example/install",
    deps = [
        "//vendor/k8s.io/apimachinery/pkg/api/testing/roundtrip:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/apis/example/fuzzer:go_default_library",
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
