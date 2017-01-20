package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "iptables.go",
        "save_restore.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/util/dbus:go_default_library",
        "//pkg/util/exec:go_default_library",
        "//pkg/util/version:go_default_library",
        "//vendor:github.com/godbus/dbus",
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["iptables_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/util/dbus:go_default_library",
        "//pkg/util/exec:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
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
        "//pkg/util/iptables/testing:all-srcs",
    ],
    tags = ["automanaged"],
)
