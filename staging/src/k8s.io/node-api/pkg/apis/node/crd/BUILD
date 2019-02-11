load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_test(
    name = "go_default_test",
    srcs = ["crd_test.go"],
    data = [
        ":manifests",
        "//cluster/addons:addon-srcs",
    ],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/apis/apiextensions:go_default_library",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1:go_default_library",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/validation:go_default_library",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/apiserver:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/validation/field:go_default_library",
        "//staging/src/k8s.io/node-api/pkg/apis/node/v1alpha1:go_default_library",
        "//vendor/github.com/stretchr/testify/assert:go_default_library",
        "//vendor/github.com/stretchr/testify/require:go_default_library",
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

filegroup(
    name = "manifests",
    srcs = ["runtimeclass_crd.yaml"],
    visibility = ["//visibility:private"],
)

go_library(
    name = "go_default_library",
    srcs = [
        "bindata.go",
        "crd.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/node-api/pkg/apis/node/crd",
    importpath = "k8s.io/node-api/pkg/apis/node/crd",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/api/legacyscheme:go_default_library",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
    ],
)
