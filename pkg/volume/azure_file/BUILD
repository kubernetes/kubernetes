package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "azure_file.go",
        "azure_provision.go",
        "azure_util.go",
        "doc.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/volume/azure_file",
    deps = [
        "//pkg/features:go_default_library",
        "//pkg/util/mount:go_default_library",
        "//pkg/volume:go_default_library",
        "//pkg/volume/util:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//staging/src/k8s.io/cloud-provider:go_default_library",
        "//staging/src/k8s.io/cloud-provider/volume/helpers:go_default_library",
        "//staging/src/k8s.io/legacy-cloud-providers/azure:go_default_library",
        "//vendor/github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2018-07-01/storage:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
        "//vendor/k8s.io/utils/strings:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["azure_file_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//pkg/util/mount:go_default_library",
        "//pkg/volume:go_default_library",
        "//pkg/volume/testing:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes/fake:go_default_library",
        "//staging/src/k8s.io/cloud-provider/fake:go_default_library",
        "//staging/src/k8s.io/legacy-cloud-providers/azure:go_default_library",
        "//vendor/github.com/Azure/go-autorest/autorest/to:go_default_library",
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
