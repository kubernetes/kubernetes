package(default_visibility = ["//visibility:public"])

load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "openstack_common.go",
        "openstack_scale.go",
        "openstack_statefulsets.go",
        "openstack_stress.go",
        "openstack_utils.go",
        "openstack_volume_cluster_ds.go",
        "openstack_volume_datastore.go",
        "openstack_volume_disksize.go",
        "openstack_volume_fstype.go",
        "openstack_volume_master_restart.go",
        "openstack_volume_ops_storm.go",
        "openstack_volume_perf.go",
        "openstack_volume_placement.go",
        "persistent_volumes-openstack.go",
        "pv_reclaimpolicy.go",
        "pvc_label_selector.go",
    ],
    importpath = "k8s.io/kubernetes/test/e2e/storage/openstack",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/cloudprovider/providers/openstack:go_default_library",
        "//pkg/volume/util:go_default_library",
        "//test/e2e/framework:go_default_library",
        "//test/e2e/storage/utils:go_default_library",
        "//vendor/github.com/onsi/ginkgo:go_default_library",
        "//vendor/github.com/onsi/gomega:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/api/storage/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/rand:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/uuid:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes:go_default_library",
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
    visibility = ["//visibility:public"],
)
