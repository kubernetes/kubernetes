load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "auth.go",
        "auth_external.go",
        "auth_sha1.go",
        "call.go",
        "conn.go",
        "conn_darwin.go",
        "conn_other.go",
        "dbus.go",
        "decoder.go",
        "doc.go",
        "encoder.go",
        "export.go",
        "homedir.go",
        "homedir_dynamic.go",
        "message.go",
        "object.go",
        "sig.go",
        "transport_darwin.go",
        "transport_generic.go",
        "transport_unix.go",
        "transport_unixcred_dragonfly.go",
        "transport_unixcred_linux.go",
        "variant.go",
        "variant_lexer.go",
        "variant_parser.go",
    ],
    cgo = True,
    importmap = "k8s.io/kubernetes/vendor/github.com/godbus/dbus",
    importpath = "github.com/godbus/dbus",
    visibility = ["//visibility:public"],
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
