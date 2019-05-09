package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "accessors.go",
        "doc.go",
        "fake.go",
        "util.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/securitycontext",
    deps = [
        "//pkg/apis/core:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "accessors_test.go",
        "util_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//pkg/apis/core:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/diff:go_default_library",
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
