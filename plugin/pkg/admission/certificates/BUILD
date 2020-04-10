load("@io_bazel_rules_go//go:def.bzl", "go_library")

package(default_visibility = ["//visibility:public"])

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
        "//plugin/pkg/admission/certificates/approval:all-srcs",
        "//plugin/pkg/admission/certificates/signing:all-srcs",
        "//plugin/pkg/admission/certificates/subjectrestriction:all-srcs",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "go_default_library",
    srcs = ["util.go"],
    importpath = "k8s.io/kubernetes/plugin/pkg/admission/certificates",
    deps = [
        "//staging/src/k8s.io/apiserver/pkg/authentication/user:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authorization/authorizer:go_default_library",
        "//vendor/k8s.io/klog/v2:go_default_library",
    ],
)
