load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "csi.go",
        "csi_objects.go",
        "in_tree.go",
    ],
    importpath = "k8s.io/kubernetes/test/e2e/storage/drivers",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/api/rbac/v1:go_default_library",
        "//staging/src/k8s.io/api/storage/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1/unstructured:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/uuid:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authentication/serviceaccount:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//test/e2e/framework:go_default_library",
        "//test/e2e/framework/auth:go_default_library",
        "//test/e2e/framework/node:go_default_library",
        "//test/e2e/framework/pod:go_default_library",
        "//test/e2e/framework/pv:go_default_library",
        "//test/e2e/framework/skipper:go_default_library",
        "//test/e2e/framework/volume:go_default_library",
        "//test/e2e/storage/testpatterns:go_default_library",
        "//test/e2e/storage/testsuites:go_default_library",
        "//test/e2e/storage/utils:go_default_library",
        "//test/e2e/storage/vsphere:go_default_library",
        "//test/utils/image:go_default_library",
        "//vendor/github.com/onsi/ginkgo:go_default_library",
        "//vendor/gopkg.in/yaml.v2:go_default_library",
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
