package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["types_test.go"],
    importpath = "k8s.io/apimachinery/pkg/apimachinery",
    library = ":go_default_library",
    deps = ["//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library"],
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "types.go",
    ],
    importpath = "k8s.io/apimachinery/pkg/apimachinery",
    deps = [
        "//vendor/k8s.io/apimachinery/pkg/api/meta:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//staging/src/k8s.io/apimachinery/pkg/apimachinery/announced:all-srcs",
        "//staging/src/k8s.io/apimachinery/pkg/apimachinery/registered:all-srcs",
    ],
    tags = ["automanaged"],
)
