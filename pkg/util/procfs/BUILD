package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "procfs.go",
        "procfs_fake.go",
        "procfs_linux.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/util/errors:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["procfs_linux_test.go"],
    library = "go_default_library",
    tags = [
        "automanaged",
        "skip",
    ],
    deps = ["//vendor:github.com/stretchr/testify/assert"],
)
