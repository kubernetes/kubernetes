load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["runtimeclass_manager.go"],
    importpath = "k8s.io/kubernetes/pkg/kubelet/runtimeclass",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/node-api/pkg/client/clientset/versioned:go_default_library",
        "//staging/src/k8s.io/node-api/pkg/client/informers/externalversions:go_default_library",
        "//staging/src/k8s.io/node-api/pkg/client/listers/node/v1alpha1:go_default_library",
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
        "//pkg/kubelet/runtimeclass/testing:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)

go_test(
    name = "go_default_test",
    srcs = ["runtimeclass_manager_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//pkg/kubelet/runtimeclass/testing:go_default_library",
        "//vendor/github.com/stretchr/testify/assert:go_default_library",
        "//vendor/k8s.io/utils/pointer:go_default_library",
    ],
)
