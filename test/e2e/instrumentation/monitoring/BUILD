package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "accelerator.go",
        "cadvisor.go",
        "custom_metrics_deployments.go",
        "custom_metrics_stackdriver.go",
        "metrics_grabber.go",
        "prometheus.go",
        "stackdriver.go",
        "stackdriver_metadata_agent.go",
    ],
    importpath = "k8s.io/kubernetes/test/e2e/instrumentation/monitoring",
    deps = [
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/api/extensions/v1beta1:go_default_library",
        "//staging/src/k8s.io/api/rbac/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/selection:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//staging/src/k8s.io/client-go/discovery:go_default_library",
        "//staging/src/k8s.io/client-go/discovery/cached:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//staging/src/k8s.io/client-go/restmapper:go_default_library",
        "//staging/src/k8s.io/metrics/pkg/client/custom_metrics:go_default_library",
        "//staging/src/k8s.io/metrics/pkg/client/external_metrics:go_default_library",
        "//test/e2e/common:go_default_library",
        "//test/e2e/framework:go_default_library",
        "//test/e2e/framework/metrics:go_default_library",
        "//test/e2e/instrumentation/common:go_default_library",
        "//test/e2e/scheduling:go_default_library",
        "//test/utils/image:go_default_library",
        "//vendor/github.com/onsi/ginkgo:go_default_library",
        "//vendor/github.com/onsi/gomega:go_default_library",
        "//vendor/github.com/prometheus/common/model:go_default_library",
        "//vendor/golang.org/x/oauth2/google:go_default_library",
        "//vendor/google.golang.org/api/monitoring/v3:go_default_library",
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
