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
        "glusterfs.go",
        "glusterfs_minmax.go",
        "glusterfs_util.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/errors:go_default_library",
        "//pkg/api/resource:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/storage/v1beta1/util:go_default_library",
        "//pkg/client/clientset_generated/release_1_5:go_default_library",
        "//pkg/labels:go_default_library",
        "//pkg/registry/core/service/allocator:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/util/exec:go_default_library",
        "//pkg/util/mount:go_default_library",
        "//pkg/util/strings:go_default_library",
        "//pkg/volume:go_default_library",
        "//pkg/volume/util:go_default_library",
        "//pkg/volume/util/volumehelper:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/heketi/heketi/client/api/go-client",
        "//vendor:github.com/heketi/heketi/pkg/glusterfs/api",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "glusterfs_minmax_test.go",
        "glusterfs_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1:go_default_library",
        "//pkg/client/clientset_generated/release_1_5/fake:go_default_library",
        "//pkg/client/testing/core:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/util/exec:go_default_library",
        "//pkg/util/mount:go_default_library",
        "//pkg/util/testing:go_default_library",
        "//pkg/volume:go_default_library",
        "//pkg/volume/testing:go_default_library",
    ],
)
