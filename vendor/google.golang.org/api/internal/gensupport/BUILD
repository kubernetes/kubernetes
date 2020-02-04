load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "buffer.go",
        "doc.go",
        "json.go",
        "jsonfloat.go",
        "media.go",
        "params.go",
        "resumable.go",
        "send.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/google.golang.org/api/internal/gensupport",
    importpath = "google.golang.org/api/internal/gensupport",
    visibility = ["//vendor/google.golang.org/api:__subpackages__"],
    deps = [
        "//vendor/github.com/googleapis/gax-go/v2:go_default_library",
        "//vendor/google.golang.org/api/googleapi:go_default_library",
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
