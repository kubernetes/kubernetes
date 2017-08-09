package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "factory.go",
        "generic.go",
    ],
    deps = [
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/client-go/tools/cache:go_default_library",
        "//vendor/k8s.io/kube-gen/test/apis/testgroup:go_default_library",
        "//vendor/k8s.io/kube-gen/test/clientset/internal:go_default_library",
        "//vendor/k8s.io/kube-gen/test/informers/internalversion/internalinterfaces:go_default_library",
        "//vendor/k8s.io/kube-gen/test/informers/internalversion/testgroup:go_default_library",
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
        "//staging/src/k8s.io/kube-gen/test/informers/internalversion/internalinterfaces:all-srcs",
        "//staging/src/k8s.io/kube-gen/test/informers/internalversion/testgroup:all-srcs",
    ],
    tags = ["automanaged"],
)
