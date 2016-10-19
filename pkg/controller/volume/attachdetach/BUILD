package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_library(
    name = "go_default_library",
    srcs = ["attach_detach_controller.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/client/cache:go_default_library",
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//pkg/client/record:go_default_library",
        "//pkg/cloudprovider:go_default_library",
        "//pkg/controller/volume/attachdetach/cache:go_default_library",
        "//pkg/controller/volume/attachdetach/populator:go_default_library",
        "//pkg/controller/volume/attachdetach/reconciler:go_default_library",
        "//pkg/controller/volume/attachdetach/statusupdater:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/util/io:go_default_library",
        "//pkg/util/mount:go_default_library",
        "//pkg/util/runtime:go_default_library",
        "//pkg/volume:go_default_library",
        "//pkg/volume/util/operationexecutor:go_default_library",
        "//pkg/volume/util/volumehelper:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["attach_detach_controller_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/client/record:go_default_library",
        "//pkg/controller/informers:go_default_library",
        "//pkg/controller/volume/attachdetach/testing:go_default_library",
    ],
)
