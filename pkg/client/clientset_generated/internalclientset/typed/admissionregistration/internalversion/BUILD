package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "admissionregistration_client.go",
        "doc.go",
        "generated_expansion.go",
        "mutatingwebhookconfiguration.go",
        "validatingwebhookconfiguration.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/admissionregistration/internalversion",
    deps = [
        "//pkg/apis/admissionregistration:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/scheme:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/watch:go_default_library",
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
        "//pkg/client/clientset_generated/internalclientset/typed/admissionregistration/internalversion/fake:all-srcs",
    ],
    tags = ["automanaged"],
)
