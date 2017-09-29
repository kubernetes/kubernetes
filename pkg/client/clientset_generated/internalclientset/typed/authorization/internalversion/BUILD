package(default_visibility = ["//visibility:public"])

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
        "selfsubjectrulesreview.go",
        "selfsubjectrulesreview_expansion.go",
        "subjectaccessreview.go",
        "subjectaccessreview_expansion.go",
    ],
    deps = [
        "//pkg/apis/authorization:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/scheme:go_default_library",
        "//vendor/k8s.io/client-go/rest:go_default_library",
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
        "//pkg/client/clientset_generated/internalclientset/typed/authorization/internalversion/fake:all-srcs",
    ],
    tags = ["automanaged"],
)
