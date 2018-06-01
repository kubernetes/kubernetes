load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["snapshot_controller.go"],
    importpath = "k8s.io/kubernetes/pkg/controller/volume/snapshot",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/cloudprovider:go_default_library",
        "//pkg/controller:go_default_library",
        "//pkg/controller/volume/snapshot/cache:go_default_library",
        "//pkg/controller/volume/snapshot/populator:go_default_library",
        "//pkg/controller/volume/snapshot/reconciler:go_default_library",
        "//pkg/controller/volume/snapshot/snapshotter:go_default_library",
        "//pkg/util/goroutinemap:go_default_library",
        "//pkg/util/io:go_default_library",
        "//pkg/util/mount:go_default_library",
        "//pkg/volume:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/api/storage/v1alpha1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
        "//vendor/k8s.io/client-go/informers/storage/v1alpha1:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/scheme:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/typed/core/v1:go_default_library",
        "//vendor/k8s.io/client-go/listers/storage/v1alpha1:go_default_library",
        "//vendor/k8s.io/client-go/tools/cache:go_default_library",
        "//vendor/k8s.io/client-go/tools/record:go_default_library",
        "//vendor/k8s.io/client-go/util/workqueue:go_default_library",
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
        "//pkg/controller/volume/snapshot/cache:all-srcs",
        "//pkg/controller/volume/snapshot/populator:all-srcs",
        "//pkg/controller/volume/snapshot/reconciler:all-srcs",
        "//pkg/controller/volume/snapshot/snapshotter:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
