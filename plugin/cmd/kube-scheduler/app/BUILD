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
    srcs = ["server.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1:go_default_library",
        "//pkg/client/clientset_generated/release_1_5:go_default_library",
        "//pkg/client/clientset_generated/release_1_5/typed/core/v1:go_default_library",
        "//pkg/client/leaderelection:go_default_library",
        "//pkg/client/leaderelection/resourcelock:go_default_library",
        "//pkg/client/record:go_default_library",
        "//pkg/client/restclient:go_default_library",
        "//pkg/client/unversioned/clientcmd:go_default_library",
        "//pkg/healthz:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/util/configz:go_default_library",
        "//plugin/cmd/kube-scheduler/app/options:go_default_library",
        "//plugin/pkg/scheduler:go_default_library",
        "//plugin/pkg/scheduler/algorithmprovider:go_default_library",
        "//plugin/pkg/scheduler/api:go_default_library",
        "//plugin/pkg/scheduler/api/latest:go_default_library",
        "//plugin/pkg/scheduler/factory:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/prometheus/client_golang/prometheus",
        "//vendor:github.com/spf13/cobra",
        "//vendor:github.com/spf13/pflag",
    ],
)
