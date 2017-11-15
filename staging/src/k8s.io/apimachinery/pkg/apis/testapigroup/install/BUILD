load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["install.go"],
    importpath = "k8s.io/apimachinery/pkg/apis/testapigroup/install",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/apimachinery/announced:go_default_library",
        "//pkg/apimachinery/registered:go_default_library",
        "//pkg/apis/testapigroup:go_default_library",
        "//pkg/apis/testapigroup/v1:go_default_library",
        "//pkg/runtime:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["roundtrip_test.go"],
    importpath = "k8s.io/apimachinery/pkg/apis/testapigroup/install",
    library = ":go_default_library",
    deps = [
        "//pkg/api/testing/roundtrip:go_default_library",
        "//pkg/apis/testapigroup/fuzzer:go_default_library",
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
