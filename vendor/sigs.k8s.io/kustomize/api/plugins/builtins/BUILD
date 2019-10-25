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
        "PatchStrategicMergeTransformer.go",
        "PatchTransformer.go",
        "PrefixSuffixTransformer.go",
        "ReplicaCountTransformer.go",
        "SecretGenerator.go",
        "doc.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/kustomize/api/plugins/builtins",
    importpath = "sigs.k8s.io/kustomize/api/plugins/builtins",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/evanphx/json-patch:go_default_library",
        "//vendor/github.com/pkg/errors:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/hasher:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/ifc:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/inventory:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/kv:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/resid:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/resmap:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/resource:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/transform:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/types:go_default_library",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
