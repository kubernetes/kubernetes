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
        "handlers.go",
        "model.go",
        "routes.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/gorilla/mux",
        "//vendor:github.com/square/go-jose",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["handlers_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = ["//vendor:github.com/square/go-jose"],
)
