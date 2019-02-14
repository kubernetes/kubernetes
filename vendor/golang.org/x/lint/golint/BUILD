load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "golint.go",
        "import.go",
        "importcomment.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/lint/golint",
    importpath = "golang.org/x/lint/golint",
    visibility = ["//visibility:private"],
    deps = ["//vendor/golang.org/x/lint:go_default_library"],
)

go_binary(
    name = "golint",
    embed = [":go_default_library"],
    tags = ["manual"],
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
