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
        "atomic_writer.go",
        "device_util.go",
        "device_util_linux.go",
        "doc.go",
        "fs.go",
        "io_util.go",
        "util.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/resource:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/storage/v1beta1:go_default_library",
        "//pkg/client/clientset_generated/release_1_5:go_default_library",
        "//pkg/util/mount:go_default_library",
        "//pkg/util/sets:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "atomic_writer_test.go",
        "device_util_linux_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/util/sets:go_default_library",
        "//pkg/util/testing:go_default_library",
    ],
)
