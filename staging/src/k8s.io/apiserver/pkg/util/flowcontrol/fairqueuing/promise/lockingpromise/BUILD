load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["lockingpromise.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing/promise/lockingpromise",
    importpath = "k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing/promise/lockingpromise",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/apiserver/pkg/util/flowcontrol/counter:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing/promise:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["lockingpromise_test.go"],
    embed = [":go_default_library"],
    deps = ["//staging/src/k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing/testing/clock:go_default_library"],
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
