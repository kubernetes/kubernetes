load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["registered.go"],
    importpath = "k8s.io/apimachinery/pkg/apimachinery/registered",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/api/meta:go_default_library",
        "//pkg/apimachinery:go_default_library",
        "//pkg/runtime/schema:go_default_library",
        "//pkg/util/sets:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["registered_test.go"],
    importpath = "k8s.io/apimachinery/pkg/apimachinery/registered",
    library = ":go_default_library",
    deps = [
        "//pkg/apimachinery:go_default_library",
        "//pkg/runtime/schema:go_default_library",
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
