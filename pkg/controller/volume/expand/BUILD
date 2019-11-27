package(default_visibility = ["//visibility:public"])

load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["expand_controller.go"],
    importpath = "k8s.io/kubernetes/pkg/controller/volume/expand",
    deps = [
        "//pkg/apis/core/v1/helper:go_default_library",
        "//pkg/controller/volume/events:go_default_library",
        "//pkg/volume:go_default_library",
        "//pkg/volume/csimigration:go_default_library",
        "//pkg/volume/util:go_default_library",
        "//pkg/volume/util/operationexecutor:go_default_library",
        "//pkg/volume/util/subpath:go_default_library",
        "//pkg/volume/util/volumepathhandler:go_default_library",
        "//staging/src/k8s.io/api/authentication/v1:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//staging/src/k8s.io/client-go/informers/core/v1:go_default_library",
        "//staging/src/k8s.io/client-go/informers/storage/v1:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes/scheme:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes/typed/core/v1:go_default_library",
        "//staging/src/k8s.io/client-go/listers/core/v1:go_default_library",
        "//staging/src/k8s.io/client-go/listers/storage/v1:go_default_library",
        "//staging/src/k8s.io/client-go/tools/cache:go_default_library",
        "//staging/src/k8s.io/client-go/tools/record:go_default_library",
        "//staging/src/k8s.io/client-go/util/workqueue:go_default_library",
        "//staging/src/k8s.io/cloud-provider:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
        "//vendor/k8s.io/utils/exec:go_default_library",
        "//vendor/k8s.io/utils/mount:go_default_library",
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
        "//pkg/controller/volume/expand/cache:all-srcs",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = ["expand_controller_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//pkg/controller:go_default_library",
        "//pkg/controller/volume/attachdetach/testing:go_default_library",
        "//pkg/features:go_default_library",
        "//pkg/volume:go_default_library",
        "//pkg/volume/awsebs:go_default_library",
        "//pkg/volume/csimigration:go_default_library",
        "//pkg/volume/util/operationexecutor:go_default_library",
        "//pkg/volume/util/types:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/api/storage/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/strategicpatch:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//staging/src/k8s.io/client-go/informers:go_default_library",
        "//staging/src/k8s.io/client-go/testing:go_default_library",
        "//staging/src/k8s.io/component-base/featuregate/testing:go_default_library",
        "//staging/src/k8s.io/csi-translation-lib:go_default_library",
        "//staging/src/k8s.io/csi-translation-lib/plugins:go_default_library",
    ],
)
