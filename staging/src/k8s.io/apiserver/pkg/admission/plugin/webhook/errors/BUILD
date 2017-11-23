load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "errors.go",
        "statuserror.go",
    ],
    importpath = "k8s.io/apiserver/pkg/admission/plugin/webhook/errors",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["statuserror_test.go"],
    importpath = "k8s.io/apiserver/pkg/admission/plugin/webhook/errors",
    library = ":go_default_library",
    deps = ["//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library"],
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
