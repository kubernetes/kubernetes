package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "connection.go",
        "constants.go",
        "custom_errors.go",
        "datacenter.go",
        "datastore.go",
        "folder.go",
        "pbm.go",
        "utils.go",
        "virtualmachine.go",
        "vmoptions.go",
        "volumeoptions.go",
        "vsphere_metrics.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/legacy-cloud-providers/vsphere/vclib",
    importpath = "k8s.io/legacy-cloud-providers/vsphere/vclib",
    deps = [
        "//staging/src/k8s.io/client-go/pkg/version:go_default_library",
        "//vendor/github.com/prometheus/client_golang/prometheus:go_default_library",
        "//vendor/github.com/vmware/govmomi/find:go_default_library",
        "//vendor/github.com/vmware/govmomi/object:go_default_library",
        "//vendor/github.com/vmware/govmomi/pbm:go_default_library",
        "//vendor/github.com/vmware/govmomi/pbm/types:go_default_library",
        "//vendor/github.com/vmware/govmomi/property:go_default_library",
        "//vendor/github.com/vmware/govmomi/session:go_default_library",
        "//vendor/github.com/vmware/govmomi/sts:go_default_library",
        "//vendor/github.com/vmware/govmomi/vim25:go_default_library",
        "//vendor/github.com/vmware/govmomi/vim25/mo:go_default_library",
        "//vendor/github.com/vmware/govmomi/vim25/soap:go_default_library",
        "//vendor/github.com/vmware/govmomi/vim25/types:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
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
        "//staging/src/k8s.io/legacy-cloud-providers/vsphere/vclib/diskmanagers:all-srcs",
        "//staging/src/k8s.io/legacy-cloud-providers/vsphere/vclib/fixtures:all-srcs",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "connection_test.go",
        "datacenter_test.go",
        "datastore_test.go",
        "folder_test.go",
        "utils_test.go",
        "virtualmachine_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/legacy-cloud-providers/vsphere/vclib/fixtures:go_default_library",
        "//vendor/github.com/vmware/govmomi:go_default_library",
        "//vendor/github.com/vmware/govmomi/object:go_default_library",
        "//vendor/github.com/vmware/govmomi/simulator:go_default_library",
    ],
)
