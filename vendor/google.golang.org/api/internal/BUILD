load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "creds.go",
        "pool.go",
        "settings.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/google.golang.org/api/internal",
    importpath = "google.golang.org/api/internal",
    visibility = ["//vendor/google.golang.org/api:__subpackages__"],
    deps = [
        "//vendor/golang.org/x/oauth2:go_default_library",
        "//vendor/golang.org/x/oauth2/google:go_default_library",
        "//vendor/google.golang.org/grpc:go_default_library",
        "//vendor/google.golang.org/grpc/naming:go_default_library",
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
        "//vendor/google.golang.org/api/internal/gensupport:all-srcs",
        "//vendor/google.golang.org/api/internal/third_party/uritemplates:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
