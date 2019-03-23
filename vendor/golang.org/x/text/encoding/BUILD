load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["encoding.go"],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/text/encoding",
    importpath = "golang.org/x/text/encoding",
    visibility = ["//visibility:public"],
    deps = [
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
    srcs = [
        ":package-srcs",
        "//vendor/golang.org/x/text/encoding/charmap:all-srcs",
        "//vendor/golang.org/x/text/encoding/htmlindex:all-srcs",
        "//vendor/golang.org/x/text/encoding/internal:all-srcs",
        "//vendor/golang.org/x/text/encoding/japanese:all-srcs",
        "//vendor/golang.org/x/text/encoding/korean:all-srcs",
        "//vendor/golang.org/x/text/encoding/simplifiedchinese:all-srcs",
        "//vendor/golang.org/x/text/encoding/traditionalchinese:all-srcs",
        "//vendor/golang.org/x/text/encoding/unicode:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
