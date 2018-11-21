package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["intstr_test.go"],
    embed = [":go_default_library"],
    deps = ["//vendor/sigs.k8s.io/yaml:go_default_library"],
)

go_library(
    name = "go_default_library",
    srcs = [
        "generated.pb.go",
        "intstr.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apimachinery/pkg/util/intstr",
    importpath = "k8s.io/apimachinery/pkg/util/intstr",
    deps = [
        "//vendor/github.com/gogo/protobuf/proto:go_default_library",
        "//vendor/github.com/google/gofuzz:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
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
