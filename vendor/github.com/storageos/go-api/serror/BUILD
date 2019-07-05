load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "error_kind.go",
        "kind_lookup_map.go",
        "storageos_error.go",
        "storageoserrorkind_string.go",
        "typed_error.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/storageos/go-api/serror",
    importpath = "github.com/storageos/go-api/serror",
    visibility = ["//visibility:public"],
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
