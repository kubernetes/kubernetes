load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "dbus.go",
        "methods.go",
        "properties.go",
        "set.go",
        "subscription.go",
        "subscription_set.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/coreos/go-systemd/dbus",
    importpath = "github.com/coreos/go-systemd/dbus",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/godbus/dbus:go_default_library"],
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
