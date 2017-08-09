package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "vsphere.go",
        "vsphere_util.go",
    ],
    deps = [
        "//pkg/api/v1/helper:go_default_library",
        "//pkg/cloudprovider:go_default_library",
        "//pkg/cloudprovider/providers/vsphere/vclib:go_default_library",
        "//pkg/cloudprovider/providers/vsphere/vclib/diskmanagers:go_default_library",
        "//pkg/controller:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/github.com/vmware/govmomi:go_default_library",
        "//vendor/github.com/vmware/govmomi/object:go_default_library",
        "//vendor/github.com/vmware/govmomi/vim25:go_default_library",
        "//vendor/github.com/vmware/govmomi/vim25/mo:go_default_library",
        "//vendor/golang.org/x/net/context:go_default_library",
        "//vendor/gopkg.in/gcfg.v1:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["vsphere_test.go"],
    library = ":go_default_library",
    deps = [
        "//pkg/cloudprovider:go_default_library",
        "//pkg/cloudprovider/providers/vsphere/vclib:go_default_library",
        "//vendor/golang.org/x/net/context:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/rand:go_default_library",
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
        "//pkg/cloudprovider/providers/vsphere/vclib:all-srcs",
    ],
    tags = ["automanaged"],
)
