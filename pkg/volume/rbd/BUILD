package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "disk_manager.go",
        "doc.go",
        "rbd.go",
        "rbd_util.go",
    ],
    deps = [
        "//pkg/util/mount:go_default_library",
        "//pkg/util/node:go_default_library",
        "//pkg/util/strings:go_default_library",
        "//pkg/volume:go_default_library",
        "//pkg/volume/util:go_default_library",
        "//pkg/volume/util/volumehelper:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/uuid:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes:go_default_library",
        "//vendor/k8s.io/utils/exec:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["rbd_test.go"],
    library = ":go_default_library",
    deps = [
        "//pkg/util/mount:go_default_library",
        "//pkg/volume:go_default_library",
        "//pkg/volume/testing:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/fake:go_default_library",
        "//vendor/k8s.io/client-go/util/testing:go_default_library",
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
