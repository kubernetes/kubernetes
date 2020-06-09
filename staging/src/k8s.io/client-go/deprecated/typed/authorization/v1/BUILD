load("@io_bazel_rules_go//go:def.bzl", "go_library")

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
    importmap = "k8s.io/kubernetes/vendor/k8s.io/client-go/deprecated/typed/authorization/v1",
    importpath = "k8s.io/client-go/deprecated/typed/authorization/v1",
    visibility = ["//staging/src/k8s.io/client-go/deprecated:__subpackages__"],
    deps = [
        "//staging/src/k8s.io/api/authorization/v1:go_default_library",
        "//staging/src/k8s.io/client-go/deprecated/scheme:go_default_library",
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
        "//staging/src/k8s.io/client-go/deprecated/typed/authorization/v1/fake:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
