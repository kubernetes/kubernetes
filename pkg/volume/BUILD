package(default_visibility = ["//visibility:public"])

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
        "volume_unsupported.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "volume_linux.go",
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//pkg/cloudprovider:go_default_library",
        "//pkg/util/io:go_default_library",
        "//pkg/util/mount:go_default_library",
        "//pkg/volume/util:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/fields:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/validation:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/watch:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes:go_default_library",
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
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/util/slice:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/watch:go_default_library",
    ],
)

go_test(
    name = "go_default_xtest",
    srcs = [
        "metrics_statfs_test.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "metrics_du_test.go",
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":go_default_library",
        "//pkg/volume/testing:go_default_library",
        "//vendor/k8s.io/client-go/util/testing:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "//conditions:default": [],
    }),
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
        "//pkg/volume/local:all-srcs",
        "//pkg/volume/nfs:all-srcs",
        "//pkg/volume/photon_pd:all-srcs",
        "//pkg/volume/portworx:all-srcs",
        "//pkg/volume/projected:all-srcs",
        "//pkg/volume/quobyte:all-srcs",
        "//pkg/volume/rbd:all-srcs",
        "//pkg/volume/scaleio:all-srcs",
        "//pkg/volume/secret:all-srcs",
        "//pkg/volume/storageos:all-srcs",
        "//pkg/volume/testing:all-srcs",
        "//pkg/volume/util:all-srcs",
        "//pkg/volume/validation:all-srcs",
        "//pkg/volume/vsphere_volume:all-srcs",
    ],
    tags = ["automanaged"],
)
