load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["protocol.go"],
    importpath = "k8s.io/apiserver/pkg/authentication/request/websocket",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/authentication/authenticator:go_default_library",
        "//pkg/authentication/user:go_default_library",
        "//pkg/util/wsstream:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["protocol_test.go"],
    importpath = "k8s.io/apiserver/pkg/authentication/request/websocket",
    library = ":go_default_library",
    deps = [
        "//pkg/authentication/authenticator:go_default_library",
        "//pkg/authentication/user:go_default_library",
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
