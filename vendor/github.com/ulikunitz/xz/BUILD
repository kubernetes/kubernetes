load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "bits.go",
        "crc.go",
        "format.go",
        "lzmafilter.go",
        "reader.go",
        "writer.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/ulikunitz/xz",
    importpath = "github.com/ulikunitz/xz",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/ulikunitz/xz/internal/xlog:go_default_library",
        "//vendor/github.com/ulikunitz/xz/lzma:go_default_library",
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
        "//vendor/github.com/ulikunitz/xz/internal/hash:all-srcs",
        "//vendor/github.com/ulikunitz/xz/internal/xlog:all-srcs",
        "//vendor/github.com/ulikunitz/xz/lzma:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
