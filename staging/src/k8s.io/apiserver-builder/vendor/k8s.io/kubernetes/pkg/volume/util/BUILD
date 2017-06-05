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
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/storage/v1beta1:go_default_library",
        "//pkg/client/clientset_generated/clientset:go_default_library",
        "//pkg/util/mount:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/api/resource",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "atomic_writer_test.go",
        "device_util_linux_test.go",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/client-go/util/testing",
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
        "//pkg/volume/util/nestedpendingoperations:all-srcs",
        "//pkg/volume/util/operationexecutor:all-srcs",
        "//pkg/volume/util/types:all-srcs",
        "//pkg/volume/util/volumehelper:all-srcs",
    ],
    tags = ["automanaged"],
)
