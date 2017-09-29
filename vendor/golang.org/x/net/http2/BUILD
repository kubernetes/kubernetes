load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "ciphers.go",
        "client_conn_pool.go",
        "configure_transport.go",
        "databuffer.go",
        "errors.go",
        "flow.go",
        "frame.go",
        "go16.go",
        "go17.go",
        "go17_not18.go",
        "go18.go",
        "go19.go",
        "gotrack.go",
        "headermap.go",
        "http2.go",
        "not_go16.go",
        "not_go17.go",
        "not_go18.go",
        "not_go19.go",
        "pipe.go",
        "server.go",
        "transport.go",
        "write.go",
        "writesched.go",
        "writesched_priority.go",
        "writesched_random.go",
    ],
    visibility = ["//visibility:public"],
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
    visibility = ["//visibility:public"],
)
