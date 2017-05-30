package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "pd.go",
        "persistent_volumes.go",
        "persistent_volumes-disruptive.go",
        "persistent_volumes-gce.go",
        "persistent_volumes-local.go",
        "persistent_volumes-vsphere.go",
        "pv_reclaimpolicy.go",
        "pvc_label_selector.go",
        "volume_provisioning.go",
        "volumes.go",
        "vsphere_utils.go",
        "vsphere_volume_diskformat.go",
        "vsphere_volume_fstype.go",
        "vsphere_volume_ops_storm.go",
        "vsphere_volume_placement.go",
        "vsphere_volume_vsan_policy.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/api/v1/helper:go_default_library",
        "//pkg/apis/rbac/v1beta1:go_default_library",
        "//pkg/apis/storage/v1:go_default_library",
        "//pkg/apis/storage/v1/util:go_default_library",
        "//pkg/apis/storage/v1beta1:go_default_library",
        "//pkg/client/clientset_generated/clientset:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/core/v1:go_default_library",
        "//pkg/cloudprovider/providers/vsphere:go_default_library",
        "//pkg/kubelet/apis:go_default_library",
        "//pkg/volume/util/volumehelper:go_default_library",
        "//test/e2e/framework:go_default_library",
        "//vendor/github.com/aws/aws-sdk-go/aws:go_default_library",
        "//vendor/github.com/aws/aws-sdk-go/aws/session:go_default_library",
        "//vendor/github.com/aws/aws-sdk-go/service/ec2:go_default_library",
        "//vendor/github.com/onsi/ginkgo:go_default_library",
        "//vendor/github.com/onsi/gomega:go_default_library",
        "//vendor/github.com/vmware/govmomi/find:go_default_library",
        "//vendor/github.com/vmware/govmomi/vim25/types:go_default_library",
        "//vendor/golang.org/x/net/context:go_default_library",
        "//vendor/google.golang.org/api/googleapi:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/uuid:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/authentication/serviceaccount:go_default_library",
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
