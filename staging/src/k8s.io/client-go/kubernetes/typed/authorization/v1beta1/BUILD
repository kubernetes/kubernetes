package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "authorization_client.go",
        "doc.go",
        "generated_expansion.go",
        "localsubjectaccessreview.go",
        "localsubjectaccessreview_expansion.go",
        "selfsubjectaccessreview.go",
        "selfsubjectaccessreview_expansion.go",
        "subjectaccessreview.go",
        "subjectaccessreview_expansion.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor/k8s.io/apimachinery/pkg/runtime/serializer:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/scheme:go_default_library",
        "//vendor/k8s.io/client-go/pkg/apis/authorization/v1beta1:go_default_library",
        "//vendor/k8s.io/client-go/rest:go_default_library",
    ],
)
