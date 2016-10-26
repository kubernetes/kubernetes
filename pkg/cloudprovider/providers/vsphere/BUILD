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
    srcs = ["vsphere.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/cloudprovider:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/util/runtime:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/vmware/govmomi",
        "//vendor:github.com/vmware/govmomi/find",
        "//vendor:github.com/vmware/govmomi/object",
        "//vendor:github.com/vmware/govmomi/property",
        "//vendor:github.com/vmware/govmomi/session",
        "//vendor:github.com/vmware/govmomi/vim25",
        "//vendor:github.com/vmware/govmomi/vim25/mo",
        "//vendor:github.com/vmware/govmomi/vim25/soap",
        "//vendor:github.com/vmware/govmomi/vim25/types",
        "//vendor:golang.org/x/net/context",
        "//vendor:gopkg.in/gcfg.v1",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["vsphere_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/cloudprovider:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/util/rand:go_default_library",
        "//vendor:golang.org/x/net/context",
    ],
)
