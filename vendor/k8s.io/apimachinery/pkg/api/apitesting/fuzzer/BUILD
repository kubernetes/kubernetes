package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["valuefuzz_test.go"],
    embed = [":go_default_library"],
)

go_library(
    name = "go_default_library",
    srcs = [
        "fuzzer.go",
        "valuefuzz.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apimachinery/pkg/api/apitesting/fuzzer",
    importpath = "k8s.io/apimachinery/pkg/api/apitesting/fuzzer",
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/runtime/serializer:go_default_library",
        "//vendor/github.com/google/gofuzz:go_default_library",
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
