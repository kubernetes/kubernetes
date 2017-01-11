package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["vsphere.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1:go_default_library",
        "//pkg/cloudprovider:go_default_library",
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
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/runtime",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["vsphere_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/cloudprovider:go_default_library",
        "//pkg/util/rand:go_default_library",
        "//vendor:golang.org/x/net/context",
        "//vendor:k8s.io/apimachinery/pkg/types",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
)
