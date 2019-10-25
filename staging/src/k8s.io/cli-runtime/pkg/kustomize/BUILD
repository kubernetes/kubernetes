load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["builder.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/cli-runtime/pkg/kustomize",
    importpath = "k8s.io/cli-runtime/pkg/kustomize",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/sigs.k8s.io/kustomize/api/filesys:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/k8sdeps/kunstruct:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/k8sdeps/transformer:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/k8sdeps/validator:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/loader:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/plugins/builtins:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/plugins/config:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/plugins/loader:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/resmap:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/resource:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/target:go_default_library",
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

go_test(
    name = "go_default_test",
    srcs = ["builder_test.go"],
    embed = [":go_default_library"],
    deps = ["//vendor/sigs.k8s.io/kustomize/api/filesys:go_default_library"],
)
