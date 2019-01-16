load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "cloudprovider.go",
        "debugging.go",
        "generic.go",
        "globalflags.go",
        "kubecloudshared.go",
        "servicecontroller.go",
    ],
    importpath = "k8s.io/kubernetes/cmd/controller-manager/app/options",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/client/leaderelectionconfig:go_default_library",
        "//pkg/cloudprovider/providers:go_default_library",
        "//pkg/controller/apis/config:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/flag:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/globalflag:go_default_library",
        "//staging/src/k8s.io/component-base/config:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
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
    visibility = ["//visibility:public"],
)

go_test(
    name = "go_default_test",
    srcs = ["globalflags_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/apiserver/pkg/util/flag:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/globalflag:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
    ],
)
