package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "globalflags.go",
        "globalflags_providers.go",
        "options.go",
        "validation.go",
    ],
    importpath = "k8s.io/kubernetes/cmd/kube-apiserver/app/options",
    deps = [
        "//pkg/api/legacyscheme:go_default_library",
        "//pkg/apis/core:go_default_library",
        "//pkg/cloudprovider/providers:go_default_library",
        "//pkg/features:go_default_library",
        "//pkg/kubeapiserver/options:go_default_library",
        "//pkg/kubelet/client:go_default_library",
        "//pkg/master/ports:go_default_library",
        "//pkg/master/reconcilers:go_default_library",
        "//pkg/serviceaccount:go_default_library",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/apiserver:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/net:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/features:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/server/options:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/storage/storagebackend:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//staging/src/k8s.io/component-base/cli/flag:go_default_library",
        "//staging/src/k8s.io/component-base/cli/globalflag:go_default_library",
        "//staging/src/k8s.io/component-base/metrics:go_default_library",
        "//staging/src/k8s.io/kube-aggregator/pkg/apiserver/scheme:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
        "//vendor/k8s.io/utils/net:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "globalflags_test.go",
        "options_test.go",
        "validation_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//pkg/apis/core:go_default_library",
        "//pkg/features:go_default_library",
        "//pkg/kubeapiserver/options:go_default_library",
        "//pkg/kubelet/client:go_default_library",
        "//pkg/master/reconcilers:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/server/options:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/storage/storagebackend:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//staging/src/k8s.io/apiserver/plugin/pkg/audit/buffered:go_default_library",
        "//staging/src/k8s.io/apiserver/plugin/pkg/audit/dynamic:go_default_library",
        "//staging/src/k8s.io/apiserver/plugin/pkg/audit/truncate:go_default_library",
        "//staging/src/k8s.io/client-go/rest:go_default_library",
        "//staging/src/k8s.io/component-base/cli/flag:go_default_library",
        "//staging/src/k8s.io/component-base/cli/globalflag:go_default_library",
        "//staging/src/k8s.io/component-base/featuregate/testing:go_default_library",
        "//staging/src/k8s.io/component-base/metrics:go_default_library",
        "//vendor/github.com/google/go-cmp/cmp:go_default_library",
        "//vendor/github.com/google/go-cmp/cmp/cmpopts:go_default_library",
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
)
