package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["doc.go"],
    importpath = "k8s.io/kubernetes/test/integration/metrics",
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

go_test(
    name = "go_default_test",
    size = "large",
    srcs = [
        "main_test.go",
        "metrics_test.go",
    ],
    embed = [":go_default_library"],
    tags = ["integration"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//staging/src/k8s.io/client-go/rest:go_default_library",
        "//test/integration/framework:go_default_library",
        "//vendor/github.com/golang/protobuf/proto:go_default_library",
        "//vendor/github.com/prometheus/client_model/go:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
    ],
)
