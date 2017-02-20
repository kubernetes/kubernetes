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
        "stateful_pod_control.go",
        "stateful_set.go",
        "stateful_set_control.go",
        "stateful_set_utils.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1/pod:go_default_library",
        "//pkg/client/clientset_generated/clientset:go_default_library",
        "//pkg/client/legacylisters:go_default_library",
        "//pkg/controller:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "stateful_pod_control_test.go",
        "stateful_set_control_test.go",
        "stateful_set_test.go",
        "stateful_set_utils_test.go",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1/pod:go_default_library",
        "//pkg/client/clientset_generated/clientset/fake:go_default_library",
        "//pkg/client/legacylisters:go_default_library",
        "//pkg/controller:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
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
