package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_test",
)

go_test(
    name = "go_default_test",
    size = "large",
    srcs = [
        "main_test.go",
        "serving_test.go",
    ],
    tags = [
        "etcd",
        "integration",
    ],
    deps = [
        "//cmd/cloud-controller-manager/app/testing:go_default_library",
        "//cmd/kube-apiserver/app/testing:go_default_library",
        "//cmd/kube-controller-manager/app/testing:go_default_library",
        "//cmd/kube-scheduler/app/testing:go_default_library",
        "//pkg/cloudprovider/providers/fake:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/server:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/server/options:go_default_library",
        "//staging/src/k8s.io/cloud-provider:go_default_library",
        "//test/integration/framework:go_default_library",
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
