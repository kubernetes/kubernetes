load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["requestheader.go"],
    importpath = "k8s.io/apiserver/pkg/authentication/request/headerrequest",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/authentication/authenticator:go_default_library",
        "//pkg/authentication/request/x509:go_default_library",
        "//pkg/authentication/user:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//vendor/k8s.io/client-go/util/cert:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["requestheader_test.go"],
    importpath = "k8s.io/apiserver/pkg/authentication/request/headerrequest",
    library = ":go_default_library",
    deps = ["//pkg/authentication/user:go_default_library"],
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
