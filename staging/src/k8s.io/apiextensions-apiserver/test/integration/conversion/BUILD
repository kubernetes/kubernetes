package(default_visibility = ["//visibility:public"])

load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_test(
    name = "go_default_test",
    srcs = ["conversion_test.go"],
    embed = [":go_default_library"],
    tags = ["integration"],
    deps = [
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1:go_default_library",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1:go_default_library",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset:go_default_library",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/cmd/server/options:go_default_library",
        "//staging/src/k8s.io/apiextensions-apiserver/test/integration/fixtures:go_default_library",
        "//staging/src/k8s.io/apiextensions-apiserver/test/integration/storage:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1/unstructured:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/uuid:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/storage/etcd3:go_default_library",
        "//staging/src/k8s.io/client-go/dynamic:go_default_library",
        "//vendor/github.com/google/go-cmp/cmp:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = ["webhook.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apiextensions-apiserver/test/integration/conversion",
    importpath = "k8s.io/apiextensions-apiserver/test/integration/conversion",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1:go_default_library",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
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
