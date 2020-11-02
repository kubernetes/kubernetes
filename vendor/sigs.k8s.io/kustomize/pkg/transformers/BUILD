load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "image.go",
        "labelsandannotations.go",
        "multitransformer.go",
        "mutatefield.go",
        "namereference.go",
        "namespace.go",
        "nooptransformer.go",
        "prefixsuffixname.go",
        "refvars.go",
        "transformer.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/kustomize/pkg/transformers",
    importpath = "sigs.k8s.io/kustomize/pkg/transformers",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/sigs.k8s.io/kustomize/pkg/expansion:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/gvk:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/image:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/resmap:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/resource:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/transformers/config:go_default_library",
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
        "//vendor/sigs.k8s.io/kustomize/pkg/transformers/config:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
