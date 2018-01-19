load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["validation.go"],
    importpath = "k8s.io/kubernetes/plugin/pkg/admission/initialresources/apis/initialresources/validation",
    visibility = ["//visibility:public"],
    deps = [
        "//plugin/pkg/admission/initialresources/apis/initialresources:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/validation/field:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["validation_test.go"],
    importpath = "k8s.io/kubernetes/plugin/pkg/admission/initialresources/apis/initialresources/validation",
    library = ":go_default_library",
    deps = ["//plugin/pkg/admission/initialresources/apis/initialresources:go_default_library"],
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
