package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["requestheader_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = ["//vendor:k8s.io/apiserver/pkg/authentication/user"],
)

go_library(
    name = "go_default_library",
    srcs = ["requestheader.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apiserver/pkg/authentication/authenticator",
        "//vendor:k8s.io/apiserver/pkg/authentication/request/x509",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
        "//vendor:k8s.io/client-go/pkg/util/cert",
        "//vendor:k8s.io/client-go/pkg/util/sets",
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
)
