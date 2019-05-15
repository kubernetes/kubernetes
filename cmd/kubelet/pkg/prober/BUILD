package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "prober.go",
        "prober_manager.go",
        "worker.go",
    ],
    importpath = "k8s.io/kubernetes/cmd/kubelet/pkg/prober",
    deps = [
        "//cmd/kubelet/pkg/container:go_default_library",
        "//cmd/kubelet/pkg/events:go_default_library",
        "//cmd/kubelet/pkg/prober/results:go_default_library",
        "//cmd/kubelet/pkg/status:go_default_library",
        "//cmd/kubelet/pkg/util/format:go_default_library",
        "//pkg/api/v1/pod:go_default_library",
        "//pkg/probe:go_default_library",
        "//pkg/probe/exec:go_default_library",
        "//pkg/probe/http:go_default_library",
        "//pkg/probe/tcp:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/intstr:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//staging/src/k8s.io/client-go/tools/record:go_default_library",
        "//vendor/github.com/prometheus/client_golang/prometheus:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
        "//vendor/k8s.io/utils/exec:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "common_test.go",
        "prober_manager_test.go",
        "prober_test.go",
        "worker_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//cmd/kubelet/pkg/container:go_default_library",
        "//cmd/kubelet/pkg/container/testing:go_default_library",
        "//cmd/kubelet/pkg/pod:go_default_library",
        "//cmd/kubelet/pkg/prober/results:go_default_library",
        "//cmd/kubelet/pkg/status:go_default_library",
        "//cmd/kubelet/pkg/status/testing:go_default_library",
        "//pkg/probe:go_default_library",
        "//pkg/probe/exec:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/intstr:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes/fake:go_default_library",
        "//staging/src/k8s.io/client-go/tools/record:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
        "//vendor/k8s.io/utils/exec:go_default_library",
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
        "//cmd/kubelet/pkg/prober/results:all-srcs",
        "//cmd/kubelet/pkg/prober/testing:all-srcs",
    ],
    tags = ["automanaged"],
)
