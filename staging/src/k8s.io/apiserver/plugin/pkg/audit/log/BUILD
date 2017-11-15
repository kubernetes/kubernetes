load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["backend.go"],
    importpath = "k8s.io/apiserver/plugin/pkg/audit/log",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/apis/audit:go_default_library",
        "//pkg/audit:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["backend_test.go"],
    importpath = "k8s.io/apiserver/plugin/pkg/audit/log",
    library = ":go_default_library",
    deps = [
        "//pkg/apis/audit:go_default_library",
        "//pkg/apis/audit/install:go_default_library",
        "//pkg/apis/audit/v1beta1:go_default_library",
        "//pkg/audit:go_default_library",
        "//vendor/github.com/pborman/uuid:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apimachinery/announced:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apimachinery/registered:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
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
