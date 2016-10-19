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
        "attacher.go",
        "vsphere_volume.go",
        "vsphere_volume_util.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/resource:go_default_library",
        "//pkg/cloudprovider:go_default_library",
        "//pkg/cloudprovider/providers/vsphere:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/util/exec:go_default_library",
        "//pkg/util/keymutex:go_default_library",
        "//pkg/util/mount:go_default_library",
        "//pkg/util/strings:go_default_library",
        "//pkg/volume:go_default_library",
        "//pkg/volume/util:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "attacher_test.go",
        "vsphere_volume_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/cloudprovider/providers/vsphere:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/util/mount:go_default_library",
        "//pkg/util/testing:go_default_library",
        "//pkg/volume:go_default_library",
        "//pkg/volume/testing:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)
