package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["feature_gate_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//vendor/github.com/spf13/pflag:go_default_library",
        "//vendor/github.com/stretchr/testify/assert:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = ["feature_gate.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apiserver/pkg/util/feature",
    importpath = "k8s.io/apiserver/pkg/util/feature",
    deps = [
        "//vendor/github.com/golang/glog:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//staging/src/k8s.io/apiserver/pkg/util/feature/testing:all-srcs",
    ],
    tags = ["automanaged"],
)
