load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "AnnotationsTransformer.go",
        "ConfigMapGenerator.go",
        "HashTransformer.go",
        "ImageTagTransformer.go",
        "InventoryTransformer.go",
        "LabelTransformer.go",
        "LegacyOrderTransformer.go",
        "NamespaceTransformer.go",
        "PatchJson6902Transformer.go",
        "PrefixSuffixTransformer.go",
        "ReplicaCountTransformer.go",
        "SecretGenerator.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/kustomize/plugin/builtin",
    importpath = "sigs.k8s.io/kustomize/plugin/builtin",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/pkg/errors:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/gvk:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/hasher:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/ifc:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/image:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/inventory:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/patch/transformer:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/resid:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/resmap:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/resource:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/transformers:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/transformers/config:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/types:go_default_library",
        "//vendor/sigs.k8s.io/yaml:go_default_library",
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
        "//vendor/sigs.k8s.io/kustomize/plugin/builtin/annotationstransformer:all-srcs",
        "//vendor/sigs.k8s.io/kustomize/plugin/builtin/configmapgenerator:all-srcs",
        "//vendor/sigs.k8s.io/kustomize/plugin/builtin/hashtransformer:all-srcs",
        "//vendor/sigs.k8s.io/kustomize/plugin/builtin/imagetagtransformer:all-srcs",
        "//vendor/sigs.k8s.io/kustomize/plugin/builtin/inventorytransformer:all-srcs",
        "//vendor/sigs.k8s.io/kustomize/plugin/builtin/labeltransformer:all-srcs",
        "//vendor/sigs.k8s.io/kustomize/plugin/builtin/legacyordertransformer:all-srcs",
        "//vendor/sigs.k8s.io/kustomize/plugin/builtin/namespacetransformer:all-srcs",
        "//vendor/sigs.k8s.io/kustomize/plugin/builtin/patchjson6902transformer:all-srcs",
        "//vendor/sigs.k8s.io/kustomize/plugin/builtin/prefixsuffixtransformer:all-srcs",
        "//vendor/sigs.k8s.io/kustomize/plugin/builtin/replicacounttransformer:all-srcs",
        "//vendor/sigs.k8s.io/kustomize/plugin/builtin/secretgenerator:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
