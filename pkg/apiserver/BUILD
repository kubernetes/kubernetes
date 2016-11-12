package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "api_installer.go",
        "apiserver.go",
        "doc.go",
        "errors.go",
        "negotiate.go",
        "proxy.go",
        "resthandler.go",
        "serviceerror.go",
        "validator.go",
        "watch.go",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "api_installer_test.go",
        "apiserver_test.go",
        "errors_test.go",
        "negotiate_test.go",
        "proxy_test.go",
        "resthandler_test.go",
        "validator_test.go",
        "watch_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [],
)
