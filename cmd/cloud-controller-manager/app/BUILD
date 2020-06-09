load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "controllermanager.go",
        "core.go",
    ],
    importpath = "k8s.io/kubernetes/cmd/cloud-controller-manager/app",
    visibility = ["//visibility:public"],
    deps = [
        "//cmd/cloud-controller-manager/app/config:go_default_library",
        "//cmd/cloud-controller-manager/app/options:go_default_library",
        "//cmd/controller-manager/app:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/uuid:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/server:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/server/healthz:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//staging/src/k8s.io/client-go/tools/leaderelection:go_default_library",
        "//staging/src/k8s.io/client-go/tools/leaderelection/resourcelock:go_default_library",
        "//staging/src/k8s.io/cloud-provider:go_default_library",
        "//staging/src/k8s.io/cloud-provider/controllers/node:go_default_library",
        "//staging/src/k8s.io/cloud-provider/controllers/nodelifecycle:go_default_library",
        "//staging/src/k8s.io/cloud-provider/controllers/route:go_default_library",
        "//staging/src/k8s.io/cloud-provider/controllers/service:go_default_library",
        "//staging/src/k8s.io/component-base/cli/flag:go_default_library",
        "//staging/src/k8s.io/component-base/cli/globalflag:go_default_library",
        "//staging/src/k8s.io/component-base/configz:go_default_library",
        "//staging/src/k8s.io/component-base/term:go_default_library",
        "//staging/src/k8s.io/component-base/version:go_default_library",
        "//staging/src/k8s.io/component-base/version/verflag:go_default_library",
        "//vendor/github.com/spf13/cobra:go_default_library",
        "//vendor/k8s.io/klog/v2:go_default_library",
        "//vendor/k8s.io/utils/net:go_default_library",
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
        "//cmd/cloud-controller-manager/app/apis/config:all-srcs",
        "//cmd/cloud-controller-manager/app/config:all-srcs",
        "//cmd/cloud-controller-manager/app/options:all-srcs",
        "//cmd/cloud-controller-manager/app/testing:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
