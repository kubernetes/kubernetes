load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "ciphers.go",
        "client_conn_pool.go",
        "databuffer.go",
        "errors.go",
        "flow.go",
        "frame.go",
        "go111.go",
        "gotrack.go",
        "headermap.go",
        "http2.go",
        "not_go111.go",
        "pipe.go",
        "server.go",
        "transport.go",
        "write.go",
        "writesched.go",
        "writesched_priority.go",
        "writesched_random.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/net/http2",
    importpath = "golang.org/x/net/http2",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/golang.org/x/net/http/httpguts:go_default_library",
        "//vendor/golang.org/x/net/http2/hpack:go_default_library",
        "//vendor/golang.org/x/net/idna:go_default_library",
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
