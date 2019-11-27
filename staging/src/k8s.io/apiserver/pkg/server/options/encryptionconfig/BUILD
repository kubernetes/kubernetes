package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["config.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apiserver/pkg/server/options/encryptionconfig",
    importpath = "k8s.io/apiserver/pkg/server/options/encryptionconfig",
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/serializer:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/apis/config:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/apis/config/v1:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/server/healthz:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/storage/value:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/aes:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/identity:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/secretbox:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["config_test.go"],
    data = glob(["testdata/**"]),
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/diff:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/apis/config:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/storage/value:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope:go_default_library",
        "//vendor/github.com/google/go-cmp/cmp:go_default_library",
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
)
