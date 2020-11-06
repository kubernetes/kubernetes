load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "localsubjectaccessreview.go",
        "nonresourceattributes.go",
        "nonresourcerule.go",
        "resourceattributes.go",
        "resourcerule.go",
        "selfsubjectaccessreview.go",
        "selfsubjectaccessreviewspec.go",
        "selfsubjectrulesreview.go",
        "selfsubjectrulesreviewspec.go",
        "subjectaccessreview.go",
        "subjectaccessreviewspec.go",
        "subjectaccessreviewstatus.go",
        "subjectrulesreviewstatus.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/client-go/applyconfigurations/authorization/v1",
    importpath = "k8s.io/client-go/applyconfigurations/authorization/v1",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/api/authorization/v1:go_default_library",
        "//staging/src/k8s.io/client-go/applyconfigurations/meta/v1:go_default_library",
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
