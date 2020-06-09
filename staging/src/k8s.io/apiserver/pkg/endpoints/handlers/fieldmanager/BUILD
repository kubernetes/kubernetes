load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "buildmanagerinfo.go",
        "capmanagers.go",
        "fieldmanager.go",
        "managedfieldsupdater.go",
        "skipnonapplied.go",
        "stripmeta.go",
        "structuredmerge.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apiserver/pkg/endpoints/handlers/fieldmanager",
    importpath = "k8s.io/apiserver/pkg/endpoints/handlers/fieldmanager",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/meta:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/endpoints/handlers/fieldmanager/internal:go_default_library",
        "//vendor/k8s.io/klog/v2:go_default_library",
        "//vendor/k8s.io/kube-openapi/pkg/util/proto:go_default_library",
        "//vendor/sigs.k8s.io/structured-merge-diff/v3/fieldpath:go_default_library",
        "//vendor/sigs.k8s.io/structured-merge-diff/v3/merge:go_default_library",
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
        "//staging/src/k8s.io/apiserver/pkg/endpoints/handlers/fieldmanager/internal:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "capmanagers_test.go",
        "fieldmanager_test.go",
        "skipnonapplied_test.go",
    ],
    data = [
        "endpoints.yaml",
        "node.yaml",
        "pod.yaml",
        "//api/openapi-spec",
    ],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/equality:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/meta:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1/unstructured:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/serializer:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/endpoints/handlers/fieldmanager/internal:go_default_library",
        "//vendor/k8s.io/kube-openapi/pkg/util/proto:go_default_library",
        "//vendor/k8s.io/kube-openapi/pkg/util/proto/testing:go_default_library",
        "//vendor/sigs.k8s.io/structured-merge-diff/v3/fieldpath:go_default_library",
        "//vendor/sigs.k8s.io/structured-merge-diff/v3/merge:go_default_library",
        "//vendor/sigs.k8s.io/structured-merge-diff/v3/typed:go_default_library",
        "//vendor/sigs.k8s.io/yaml:go_default_library",
    ],
)
