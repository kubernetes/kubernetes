package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "credentialmanager.go",
        "nodemanager.go",
        "vsphere.go",
        "vsphere_util.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/cloudprovider/providers/vsphere",
    deps = [
        "//pkg/apis/core/v1/helper:go_default_library",
        "//pkg/cloudprovider:go_default_library",
        "//pkg/cloudprovider/providers/vsphere/vclib:go_default_library",
        "//pkg/cloudprovider/providers/vsphere/vclib/diskmanagers:go_default_library",
        "//pkg/controller:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/version:go_default_library",
        "//staging/src/k8s.io/client-go/informers:go_default_library",
        "//staging/src/k8s.io/client-go/listers/core/v1:go_default_library",
        "//staging/src/k8s.io/client-go/tools/cache:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/github.com/vmware/govmomi/vapi/rest:go_default_library",
        "//vendor/github.com/vmware/govmomi/vapi/tags:go_default_library",
        "//vendor/github.com/vmware/govmomi/vim25:go_default_library",
        "//vendor/github.com/vmware/govmomi/vim25/mo:go_default_library",
        "//vendor/gopkg.in/gcfg.v1:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "credentialmanager_test.go",
        "vsphere_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//pkg/cloudprovider:go_default_library",
        "//pkg/cloudprovider/providers/vsphere/vclib:go_default_library",
        "//pkg/cloudprovider/providers/vsphere/vclib/fixtures:go_default_library",
        "//pkg/controller:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/rand:go_default_library",
        "//staging/src/k8s.io/client-go/informers:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes/fake:go_default_library",
        "//vendor/github.com/vmware/govmomi/lookup/simulator:go_default_library",
        "//vendor/github.com/vmware/govmomi/property:go_default_library",
        "//vendor/github.com/vmware/govmomi/simulator:go_default_library",
        "//vendor/github.com/vmware/govmomi/simulator/vpx:go_default_library",
        "//vendor/github.com/vmware/govmomi/sts/simulator:go_default_library",
        "//vendor/github.com/vmware/govmomi/vapi/rest:go_default_library",
        "//vendor/github.com/vmware/govmomi/vapi/simulator:go_default_library",
        "//vendor/github.com/vmware/govmomi/vapi/tags:go_default_library",
        "//vendor/github.com/vmware/govmomi/vim25/mo:go_default_library",
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
