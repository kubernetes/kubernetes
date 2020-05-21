load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "add.go",
        "atomic_value.go",
        "atomic_value_go13.go",
        "bind.go",
        "client.go",
        "compare.go",
        "conn.go",
        "control.go",
        "debug.go",
        "del.go",
        "dn.go",
        "doc.go",
        "error.go",
        "filter.go",
        "ldap.go",
        "modify.go",
        "passwdmodify.go",
        "search.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/gopkg.in/ldap.v2",
    importpath = "gopkg.in/ldap.v2",
    visibility = ["//visibility:public"],
    deps = ["//vendor/gopkg.in/asn1-ber.v1:go_default_library"],
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
