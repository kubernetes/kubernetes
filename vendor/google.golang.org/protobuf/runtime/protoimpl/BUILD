load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "impl.go",
        "version.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/google.golang.org/protobuf/runtime/protoimpl",
    importpath = "google.golang.org/protobuf/runtime/protoimpl",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/google.golang.org/protobuf/internal/filedesc:go_default_library",
        "//vendor/google.golang.org/protobuf/internal/filetype:go_default_library",
        "//vendor/google.golang.org/protobuf/internal/impl:go_default_library",
        "//vendor/google.golang.org/protobuf/internal/version:go_default_library",
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
