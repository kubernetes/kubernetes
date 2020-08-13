package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["server.go"],
    importpath = "k8s.io/kubernetes/cmd/kube-scheduler/app",
    deps = [
        "//cmd/kube-scheduler/app/config:go_default_library",
        "//cmd/kube-scheduler/app/options:go_default_library",
        "//pkg/api/legacyscheme:go_default_library",
        "//pkg/scheduler:go_default_library",
        "//pkg/scheduler/apis/config:go_default_library",
        "//pkg/scheduler/framework/runtime:go_default_library",
        "//pkg/scheduler/metrics:go_default_library",
        "//pkg/scheduler/profile:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/errors:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authentication/authenticator:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authorization/authorizer:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/endpoints/filters:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/endpoints/request:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/server/filters:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/server/healthz:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/server/mux:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/server/routes:go_default_library",
        "//staging/src/k8s.io/client-go/tools/events:go_default_library",
        "//staging/src/k8s.io/client-go/tools/leaderelection:go_default_library",
        "//staging/src/k8s.io/component-base/cli/flag:go_default_library",
        "//staging/src/k8s.io/component-base/cli/globalflag:go_default_library",
        "//staging/src/k8s.io/component-base/configz:go_default_library",
        "//staging/src/k8s.io/component-base/logs:go_default_library",
        "//staging/src/k8s.io/component-base/metrics/legacyregistry:go_default_library",
        "//staging/src/k8s.io/component-base/term:go_default_library",
        "//staging/src/k8s.io/component-base/version:go_default_library",
        "//staging/src/k8s.io/component-base/version/verflag:go_default_library",
        "//vendor/github.com/spf13/cobra:go_default_library",
        "//vendor/k8s.io/klog/v2:go_default_library",
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
        "//cmd/kube-scheduler/app/config:all-srcs",
        "//cmd/kube-scheduler/app/options:all-srcs",
        "//cmd/kube-scheduler/app/testing:all-srcs",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = ["server_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//cmd/kube-scheduler/app/options:go_default_library",
        "//pkg/scheduler/apis/config:go_default_library",
        "//vendor/github.com/google/go-cmp/cmp:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
    ],
)
