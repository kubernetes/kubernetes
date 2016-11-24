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
    srcs = ["reconciler.go"],
    tags = ["automanaged"],
    deps = [
        "//cmd/kubelet/app/options:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/client/clientset_generated/release_1_5:go_default_library",
        "//pkg/kubelet/config:go_default_library",
        "//pkg/kubelet/volumemanager/cache:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/util:go_default_library",
        "//pkg/util/goroutinemap/exponentialbackoff:go_default_library",
        "//pkg/util/mount:go_default_library",
        "//pkg/util/strings:go_default_library",
        "//pkg/util/wait:go_default_library",
        "//pkg/volume:go_default_library",
        "//pkg/volume/util/nestedpendingoperations:go_default_library",
        "//pkg/volume/util/operationexecutor:go_default_library",
        "//pkg/volume/util/types:go_default_library",
        "//pkg/volume/util/volumehelper:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["reconciler_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1:go_default_library",
        "//pkg/client/clientset_generated/release_1_5/fake:go_default_library",
        "//pkg/client/record:go_default_library",
        "//pkg/client/testing/core:go_default_library",
        "//pkg/kubelet/config:go_default_library",
        "//pkg/kubelet/volumemanager/cache:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/util/mount:go_default_library",
        "//pkg/util/sets:go_default_library",
        "//pkg/util/wait:go_default_library",
        "//pkg/volume:go_default_library",
        "//pkg/volume/testing:go_default_library",
        "//pkg/volume/util/operationexecutor:go_default_library",
        "//pkg/volume/util/volumehelper:go_default_library",
        "//vendor:github.com/stretchr/testify/assert",
    ],
)
