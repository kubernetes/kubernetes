load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "cloudprovider.go",
        "kubecloudshared.go",
        "options.go",
        "servicecontroller.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/cloud-provider/options",
    importpath = "k8s.io/cloud-provider/options",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/errors:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/server/options:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//staging/src/k8s.io/client-go/informers:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes/scheme:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes/typed/core/v1:go_default_library",
        "//staging/src/k8s.io/client-go/rest:go_default_library",
        "//staging/src/k8s.io/client-go/tools/clientcmd:go_default_library",
        "//staging/src/k8s.io/client-go/tools/record:go_default_library",
        "//staging/src/k8s.io/cloud-provider:go_default_library",
        "//staging/src/k8s.io/cloud-provider/app/config:go_default_library",
        "//staging/src/k8s.io/cloud-provider/config:go_default_library",
        "//staging/src/k8s.io/cloud-provider/config/install:go_default_library",
        "//staging/src/k8s.io/cloud-provider/config/v1alpha1:go_default_library",
        "//staging/src/k8s.io/cloud-provider/controllers/service/config:go_default_library",
        "//staging/src/k8s.io/component-base/cli/flag:go_default_library",
        "//staging/src/k8s.io/controller-manager/options:go_default_library",
        "//staging/src/k8s.io/controller-manager/pkg/clientbuilder:go_default_library",
        "//staging/src/k8s.io/controller-manager/pkg/features/register:go_default_library",
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
    srcs = ["options_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/diff:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/server/options:go_default_library",
        "//staging/src/k8s.io/cloud-provider/config:go_default_library",
        "//staging/src/k8s.io/cloud-provider/controllers/service/config:go_default_library",
        "//staging/src/k8s.io/component-base/config:go_default_library",
        "//staging/src/k8s.io/controller-manager/config:go_default_library",
        "//staging/src/k8s.io/controller-manager/options:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
    ],
)
