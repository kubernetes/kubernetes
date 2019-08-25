load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["fakeloader.go"],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/kustomize/internal/loadertest",
    importpath = "sigs.k8s.io/kustomize/internal/loadertest",
    visibility = ["//vendor/sigs.k8s.io/kustomize:__subpackages__"],
    deps = [
        "//vendor/sigs.k8s.io/kustomize/pkg/fs:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/ifc:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/loader:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/types:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/validators:go_default_library",
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
    visibility = ["//visibility:public"],
)
