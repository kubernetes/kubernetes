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
        "cinder.go",
        "cinder_util.go",
        "doc.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/resource:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/cloudprovider:go_default_library",
        "//pkg/cloudprovider/providers/openstack:go_default_library",
        "//pkg/cloudprovider/providers/rackspace:go_default_library",
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
        "cinder_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1:go_default_library",
        "//pkg/cloudprovider:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/util/mount:go_default_library",
        "//pkg/util/testing:go_default_library",
        "//pkg/volume:go_default_library",
        "//pkg/volume/testing:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)
