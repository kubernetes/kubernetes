load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "htmlindex.go",
        "map.go",
        "tables.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/text/encoding/htmlindex",
    importpath = "golang.org/x/text/encoding/htmlindex",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/golang.org/x/text/encoding:go_default_library",
        "//vendor/golang.org/x/text/encoding/charmap:go_default_library",
        "//vendor/golang.org/x/text/encoding/internal/identifier:go_default_library",
        "//vendor/golang.org/x/text/encoding/japanese:go_default_library",
        "//vendor/golang.org/x/text/encoding/korean:go_default_library",
        "//vendor/golang.org/x/text/encoding/simplifiedchinese:go_default_library",
        "//vendor/golang.org/x/text/encoding/traditionalchinese:go_default_library",
        "//vendor/golang.org/x/text/encoding/unicode:go_default_library",
        "//vendor/golang.org/x/text/language:go_default_library",
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
