load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "all.go",
        "gbk.go",
        "hzgb2312.go",
        "tables.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/text/encoding/simplifiedchinese",
    importpath = "golang.org/x/text/encoding/simplifiedchinese",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/golang.org/x/text/encoding:go_default_library",
        "//vendor/golang.org/x/text/encoding/internal:go_default_library",
        "//vendor/golang.org/x/text/encoding/internal/identifier:go_default_library",
        "//vendor/golang.org/x/text/transform:go_default_library",
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
