load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["external.go"],
    importpath = "k8s.io/kubernetes/test/e2e/storage/external",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/api/storage/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1/unstructured:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/storage/names:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes/scheme:go_default_library",
        "//test/e2e/framework:go_default_library",
        "//test/e2e/framework/config:go_default_library",
        "//test/e2e/framework/skipper:go_default_library",
        "//test/e2e/framework/volume:go_default_library",
        "//test/e2e/storage/testpatterns:go_default_library",
        "//test/e2e/storage/testsuites:go_default_library",
        "//test/e2e/storage/utils:go_default_library",
        "//vendor/github.com/onsi/ginkgo:go_default_library",
        "//vendor/github.com/pkg/errors:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["external_test.go"],
    data = glob(["testdata/**"]),
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//test/e2e/framework/volume:go_default_library",
        "//test/e2e/storage/testsuites:go_default_library",
        "//vendor/github.com/stretchr/testify/assert:go_default_library",
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
