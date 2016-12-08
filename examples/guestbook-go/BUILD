package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
)

go_binary(
    name = "guestbook-go",
    srcs = ["main.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/codegangsta/negroni",
        "//vendor:github.com/gorilla/mux",
        "//vendor:github.com/xyproto/simpleredis",
    ],
)
