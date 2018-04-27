load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "auth.go",
        "auth_external.go",
        "auth_sha1.go",
        "call.go",
        "conn.go",
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
        "transport_generic.go",
        "variant.go",
        "variant_lexer.go",
        "variant_parser.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "conn_other.go",
            "transport_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "conn_darwin.go",
            "transport_darwin.go",
            "transport_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "conn_other.go",
            "transport_unix.go",
            "transport_unixcred_dragonfly.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "conn_other.go",
            "transport_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "conn_other.go",
            "transport_unix.go",
            "transport_unixcred_linux.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "conn_other.go",
            "transport_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "conn_other.go",
            "transport_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "conn_other.go",
            "transport_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "conn_other.go",
            "transport_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "conn_other.go",
            "transport_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "conn_other.go",
        ],
        "//conditions:default": [],
    }),
    cgo = True,
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
