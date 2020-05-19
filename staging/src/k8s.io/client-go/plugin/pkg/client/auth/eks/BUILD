package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["eks_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//vendor/github.com/aws/aws-sdk-go/aws:go_default_library"
        "//vendor/github.com/aws/aws-sdk-go/aws/credentials:go_default_library"
        "//vendor/github.com/aws/aws-sdk-go/aws/credentials/stscreds:go_default_library"
        "//vendor/github.com/aws/aws-sdk-go/aws/session:go_default_library"
        "//vendor/github.com/aws/aws-sdk-go/aws/signer/v4:go_default_library"
    ],
)

go_library(
    name = "go_default_library",
    srcs = ["eks.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/client-go/plugin/pkg/client/auth/eks",
    importpath = "k8s.io/client-go/plugin/pkg/client/auth/eks",
    deps = [
        "//staging/src/k8s.io/client-go/rest:go_default_library",
        "//vendor/github.com/aws/aws-sdk-go/aws:go_default_library"
        "//vendor/github.com/aws/aws-sdk-go/aws/credentials:go_default_library"
        "//vendor/github.com/aws/aws-sdk-go/aws/credentials/stscreds:go_default_library"
        "//vendor/github.com/aws/aws-sdk-go/aws/session:go_default_library"
        "//vendor/github.com/aws/aws-sdk-go/aws/signer/v4:go_default_library"
        "//vendor/k8s.io/klog/v2:go_default_library",
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
