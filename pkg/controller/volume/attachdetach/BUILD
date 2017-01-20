package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["attach_detach_controller.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/client/cache:go_default_library",
        "//pkg/client/clientset_generated/clientset:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/core/v1:go_default_library",
        "//pkg/client/record:go_default_library",
        "//pkg/cloudprovider:go_default_library",
        "//pkg/controller/volume/attachdetach/cache:go_default_library",
        "//pkg/controller/volume/attachdetach/populator:go_default_library",
        "//pkg/controller/volume/attachdetach/reconciler:go_default_library",
        "//pkg/controller/volume/attachdetach/statusupdater:go_default_library",
        "//pkg/util/io:go_default_library",
        "//pkg/util/mount:go_default_library",
        "//pkg/volume:go_default_library",
        "//pkg/volume/util/operationexecutor:go_default_library",
        "//pkg/volume/util/volumehelper:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/runtime",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["attach_detach_controller_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/controller/informers:go_default_library",
        "//pkg/controller/volume/attachdetach/testing:go_default_library",
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
        "//pkg/controller/volume/attachdetach/cache:all-srcs",
        "//pkg/controller/volume/attachdetach/populator:all-srcs",
        "//pkg/controller/volume/attachdetach/reconciler:all-srcs",
        "//pkg/controller/volume/attachdetach/statusupdater:all-srcs",
        "//pkg/controller/volume/attachdetach/testing:all-srcs",
    ],
    tags = ["automanaged"],
)
