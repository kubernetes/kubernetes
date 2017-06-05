package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["validation_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//cmd/kubeadm/app/apis/kubeadm:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/util/validation/field",
    ],
)

go_library(
    name = "go_default_library",
    srcs = ["validation.go"],
    tags = ["automanaged"],
    deps = [
        "//cmd/kubeadm/app/apis/kubeadm:go_default_library",
        "//cmd/kubeadm/app/constants:go_default_library",
        "//cmd/kubeadm/app/util/token:go_default_library",
        "//pkg/api/validation:go_default_library",
        "//pkg/kubeapiserver/authorizer/modes:go_default_library",
        "//pkg/registry/core/service/ipallocator:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/util/validation",
        "//vendor:k8s.io/apimachinery/pkg/util/validation/field",
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
