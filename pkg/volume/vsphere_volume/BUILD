package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "attacher.go",
        "doc.go",
        "vsphere_volume.go",
        "vsphere_volume_block.go",
        "vsphere_volume_util.go",
        "vsphere_volume_util_linux.go",
        "vsphere_volume_util_unsupported.go",
        "vsphere_volume_util_windows.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/volume/vsphere_volume",
    deps = [
        "//pkg/features:go_default_library",
        "//pkg/volume:go_default_library",
        "//pkg/volume/util:go_default_library",
        "//pkg/volume/util/volumepathhandler:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//staging/src/k8s.io/cloud-provider:go_default_library",
        "//staging/src/k8s.io/cloud-provider/volume/helpers:go_default_library",
        "//staging/src/k8s.io/legacy-cloud-providers/vsphere:go_default_library",
        "//staging/src/k8s.io/legacy-cloud-providers/vsphere/vclib:go_default_library",
        "//staging/src/k8s.io/mount-utils:go_default_library",
        "//vendor/k8s.io/klog/v2:go_default_library",
        "//vendor/k8s.io/utils/keymutex:go_default_library",
        "//vendor/k8s.io/utils/strings:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "attacher_test.go",
        "vsphere_volume_block_test.go",
        "vsphere_volume_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//pkg/volume:go_default_library",
        "//pkg/volume/testing:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/client-go/util/testing:go_default_library",
        "//staging/src/k8s.io/cloud-provider:go_default_library",
        "//staging/src/k8s.io/cloud-provider/fake:go_default_library",
        "//staging/src/k8s.io/legacy-cloud-providers/vsphere:go_default_library",
        "//staging/src/k8s.io/legacy-cloud-providers/vsphere/vclib:go_default_library",
        "//staging/src/k8s.io/mount-utils:go_default_library",
        "//vendor/k8s.io/klog/v2:go_default_library",
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
