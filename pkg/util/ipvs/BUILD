package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["ipvs_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/util/dbus:go_default_library",
        "//pkg/util/exec:go_default_library",
        "//vendor/github.com/docker/libnetwork/ipvs:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = ["ipvs.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/util/dbus:go_default_library",
        "//pkg/util/exec:go_default_library",
        "//pkg/util/sysctl:go_default_library",
        "//vendor/github.com/docker/libnetwork/ipvs:go_default_library",
        "//vendor/github.com/godbus/dbus:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
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
        "//pkg/util/ipvs/testing:all-srcs",
    ],
    tags = ["automanaged"],
)
