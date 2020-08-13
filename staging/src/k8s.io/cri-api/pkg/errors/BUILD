load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "errors.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/cri-api/pkg/errors",
    importpath = "k8s.io/cri-api/pkg/errors",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/google.golang.org/grpc/codes:go_default_library",
        "//vendor/google.golang.org/grpc/status:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["errors_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//vendor/google.golang.org/grpc/codes:go_default_library",
        "//vendor/google.golang.org/grpc/status:go_default_library",
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
