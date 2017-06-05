package(default_visibility = ["//visibility:public"])

licenses(["notice"])

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
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1:go_default_library",
        "//pkg/kubelet/container:go_default_library",
        "//pkg/kubelet/events:go_default_library",
        "//pkg/kubelet/prober/results:go_default_library",
        "//pkg/kubelet/status:go_default_library",
        "//pkg/kubelet/util/format:go_default_library",
        "//pkg/probe:go_default_library",
        "//pkg/probe/exec:go_default_library",
        "//pkg/probe/http:go_default_library",
        "//pkg/probe/tcp:go_default_library",
        "//pkg/util/exec:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/intstr",
        "//vendor:k8s.io/apimachinery/pkg/util/runtime",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apimachinery/pkg/util/wait",
        "//vendor:k8s.io/client-go/tools/record",
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
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1:go_default_library",
        "//pkg/client/clientset_generated/clientset/fake:go_default_library",
        "//pkg/kubelet/container:go_default_library",
        "//pkg/kubelet/container/testing:go_default_library",
        "//pkg/kubelet/pod:go_default_library",
        "//pkg/kubelet/prober/results:go_default_library",
        "//pkg/kubelet/status:go_default_library",
        "//pkg/kubelet/status/testing:go_default_library",
        "//pkg/probe:go_default_library",
        "//pkg/probe/exec:go_default_library",
        "//pkg/util/exec:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/intstr",
        "//vendor:k8s.io/apimachinery/pkg/util/runtime",
        "//vendor:k8s.io/apimachinery/pkg/util/wait",
        "//vendor:k8s.io/client-go/tools/record",
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
        "//pkg/kubelet/prober/results:all-srcs",
        "//pkg/kubelet/prober/testing:all-srcs",
    ],
    tags = ["automanaged"],
)
