load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "annotate.go",
        "retry.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/cloud.google.com/go/internal",
    importpath = "cloud.google.com/go/internal",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/googleapis/gax-go:go_default_library",
        "//vendor/golang.org/x/net/context:go_default_library",
        "//vendor/google.golang.org/api/googleapi:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//vendor/cloud.google.com/go/internal/version:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
