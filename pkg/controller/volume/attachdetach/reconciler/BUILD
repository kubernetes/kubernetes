package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["reconciler.go"],
    deps = [
        "//pkg/controller/volume/attachdetach/cache:go_default_library",
        "//pkg/controller/volume/attachdetach/statusupdater:go_default_library",
        "//pkg/kubelet/events:go_default_library",
        "//pkg/util/goroutinemap/exponentialbackoff:go_default_library",
        "//pkg/volume:go_default_library",
        "//pkg/volume/util/operationexecutor:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//vendor/k8s.io/client-go/tools/record:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["reconciler_test.go"],
    library = ":go_default_library",
    deps = [
        "//pkg/controller:go_default_library",
        "//pkg/controller/volume/attachdetach/cache:go_default_library",
        "//pkg/controller/volume/attachdetach/statusupdater:go_default_library",
        "//pkg/controller/volume/attachdetach/testing:go_default_library",
        "//pkg/volume/testing:go_default_library",
        "//pkg/volume/util/operationexecutor:go_default_library",
        "//pkg/volume/util/types:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//vendor/k8s.io/client-go/informers:go_default_library",
        "//vendor/k8s.io/client-go/tools/record:go_default_library",
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
