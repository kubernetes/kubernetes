load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "fake_apps_client.go",
        "fake_controllerrevision.go",
        "fake_daemonset.go",
        "fake_deployment.go",
        "fake_replicaset.go",
        "fake_statefulset.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/client-go/deprecated/typed/apps/v1beta2/fake",
    importpath = "k8s.io/client-go/deprecated/typed/apps/v1beta2/fake",
    visibility = ["//staging/src/k8s.io/client-go/deprecated:__subpackages__"],
    deps = [
        "//staging/src/k8s.io/api/apps/v1beta2:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/watch:go_default_library",
        "//staging/src/k8s.io/client-go/deprecated/typed/apps/v1beta2:go_default_library",
        "//staging/src/k8s.io/client-go/rest:go_default_library",
        "//staging/src/k8s.io/client-go/testing:go_default_library",
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
