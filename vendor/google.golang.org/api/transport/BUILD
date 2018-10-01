load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "dial.go",
        "go19.go",
        "not_go19.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/google.golang.org/api/transport",
    importpath = "google.golang.org/api/transport",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/golang.org/x/net/context:go_default_library",
        "//vendor/golang.org/x/oauth2/google:go_default_library",
        "//vendor/google.golang.org/api/internal:go_default_library",
        "//vendor/google.golang.org/api/option:go_default_library",
        "//vendor/google.golang.org/api/transport/grpc:go_default_library",
        "//vendor/google.golang.org/api/transport/http:go_default_library",
        "//vendor/google.golang.org/grpc:go_default_library",
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
        "//vendor/google.golang.org/api/transport/grpc:all-srcs",
        "//vendor/google.golang.org/api/transport/http:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
