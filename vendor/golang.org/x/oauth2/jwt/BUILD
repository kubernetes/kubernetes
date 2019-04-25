load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["jwt.go"],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/oauth2/jwt",
    importpath = "golang.org/x/oauth2/jwt",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/golang.org/x/oauth2:go_default_library",
        "//vendor/golang.org/x/oauth2/internal:go_default_library",
        "//vendor/golang.org/x/oauth2/jws:go_default_library",
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
