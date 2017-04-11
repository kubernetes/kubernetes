package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "metrics_cached.go",
        "metrics_du.go",
        "metrics_errors.go",
        "metrics_nil.go",
        "metrics_statfs.go",
        "plugins.go",
        "util.go",
        "volume.go",
        "volume_linux.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1:go_default_library",
        "//pkg/client/clientset_generated/clientset:go_default_library",
        "//pkg/cloudprovider:go_default_library",
        "//pkg/util/chmod:go_default_library",
        "//pkg/util/chown:go_default_library",
        "//pkg/util/io:go_default_library",
        "//pkg/util/mount:go_default_library",
        "//pkg/volume/util:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/api/resource",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/fields",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/errors",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apimachinery/pkg/util/validation",
        "//vendor:k8s.io/apimachinery/pkg/watch",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "metrics_nil_test.go",
        "plugins_test.go",
        "util_test.go",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/util/slice:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/api/resource",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apimachinery/pkg/watch",
    ],
)

go_test(
    name = "go_default_xtest",
    srcs = [
        "metrics_du_test.go",
        "metrics_statfs_test.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/volume:go_default_library",
        "//pkg/volume/testing:go_default_library",
        "//vendor:k8s.io/client-go/util/testing",
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
        "//pkg/volume/aws_ebs:all-srcs",
        "//pkg/volume/azure_dd:all-srcs",
        "//pkg/volume/azure_file:all-srcs",
        "//pkg/volume/cephfs:all-srcs",
        "//pkg/volume/cinder:all-srcs",
        "//pkg/volume/configmap:all-srcs",
        "//pkg/volume/downwardapi:all-srcs",
        "//pkg/volume/empty_dir:all-srcs",
        "//pkg/volume/fc:all-srcs",
        "//pkg/volume/flexvolume:all-srcs",
        "//pkg/volume/flocker:all-srcs",
        "//pkg/volume/gce_pd:all-srcs",
        "//pkg/volume/git_repo:all-srcs",
        "//pkg/volume/glusterfs:all-srcs",
        "//pkg/volume/host_path:all-srcs",
        "//pkg/volume/iscsi:all-srcs",
        "//pkg/volume/nfs:all-srcs",
        "//pkg/volume/photon_pd:all-srcs",
        "//pkg/volume/portworx:all-srcs",
        "//pkg/volume/projected:all-srcs",
        "//pkg/volume/quobyte:all-srcs",
        "//pkg/volume/rbd:all-srcs",
        "//pkg/volume/scaleio:all-srcs",
        "//pkg/volume/secret:all-srcs",
        "//pkg/volume/testing:all-srcs",
        "//pkg/volume/util:all-srcs",
        "//pkg/volume/validation:all-srcs",
        "//pkg/volume/vsphere_volume:all-srcs",
    ],
    tags = ["automanaged"],
)
