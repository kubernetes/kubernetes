load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "event.go",
        "event_expansion.go",
        "events_client.go",
        "generated_expansion.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/client-go/kubernetes/typed/events/v1beta1",
    importpath = "k8s.io/client-go/kubernetes/typed/events/v1beta1",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/api/events/v1beta1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/watch:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes/scheme:go_default_library",
        "//staging/src/k8s.io/client-go/rest:go_default_library",
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
        "//staging/src/k8s.io/client-go/kubernetes/typed/events/v1beta1/fake:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
