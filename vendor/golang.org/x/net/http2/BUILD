package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "client_conn_pool.go",
        "configure_transport.go",
        "errors.go",
        "fixed_buffer.go",
        "flow.go",
        "frame.go",
        "go16.go",
        "go17.go",
        "go18.go",
        "gotrack.go",
        "headermap.go",
        "http2.go",
        "pipe.go",
        "server.go",
        "transport.go",
        "write.go",
        "writesched.go",
        "writesched_priority.go",
        "writesched_random.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor/golang.org/x/net/http2/hpack:go_default_library",
        "//vendor/golang.org/x/net/idna:go_default_library",
        "//vendor/golang.org/x/net/lex/httplex:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//vendor/golang.org/x/net/http2/hpack:all-srcs",
    ],
    tags = ["automanaged"],
)
