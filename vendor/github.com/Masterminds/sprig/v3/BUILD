load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "crypto.go",
        "date.go",
        "defaults.go",
        "dict.go",
        "doc.go",
        "functions.go",
        "list.go",
        "network.go",
        "numeric.go",
        "reflect.go",
        "regex.go",
        "semver.go",
        "strings.go",
        "url.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/Masterminds/sprig/v3",
    importpath = "github.com/Masterminds/sprig/v3",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/Masterminds/goutils:go_default_library",
        "//vendor/github.com/Masterminds/semver/v3:go_default_library",
        "//vendor/github.com/google/uuid:go_default_library",
        "//vendor/github.com/huandu/xstrings:go_default_library",
        "//vendor/github.com/imdario/mergo:go_default_library",
        "//vendor/github.com/mitchellh/copystructure:go_default_library",
        "//vendor/github.com/spf13/cast:go_default_library",
        "//vendor/golang.org/x/crypto/scrypt:go_default_library",
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
