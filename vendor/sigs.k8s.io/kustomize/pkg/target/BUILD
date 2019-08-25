load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "kusttarget.go",
        "kusttarget_configplugin.go",
        "resaccumulator.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/kustomize/pkg/target",
    importpath = "sigs.k8s.io/kustomize/pkg/target",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/pkg/errors:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/accumulator:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/ifc:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/ifc/transformer:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/image:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/pgmconfig:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/plugins:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/resid:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/resmap:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/transformers:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/transformers/config:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/types:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/plugin/builtin:go_default_library",
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

go_test(
    name = "go_default_test",
    srcs = [
        "baseandoverlaymedium_test.go",
        "baseandoverlaysmall_test.go",
        "configmaps_test.go",
        "crd_test.go",
        "customconfig_test.go",
        "generatormergeandreplace_test.go",
        "generatoroptions_test.go",
        "kusttarget_test.go",
        "multiplepatch_test.go",
        "namespacedgenerators_test.go",
        "nullvalues_test.go",
        "plugindir_test.go",
        "pruneconfigmap_test.go",
        "resourceconflict_test.go",
        "transformerplugin_test.go",
        "transformersarrays_test.go",
        "transformersimage_test.go",
        "variableref_test.go",
    ],
    embed = [":go_default_library"],
    tags = ["manual"],
    deps = [
        "//vendor/sigs.k8s.io/kustomize/internal/loadertest:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/k8sdeps/kunstruct:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/k8sdeps/transformer:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/fs:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/gvk:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/ifc:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/kusttest:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/loader:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/plugins:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/resmap:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/resource:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/types:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/validators:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/plugin:go_default_library",
    ],
)
