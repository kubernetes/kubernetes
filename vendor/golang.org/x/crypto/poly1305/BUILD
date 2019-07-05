load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "poly1305.go",
        "sum_amd64.go",
        "sum_amd64.s",
        "sum_arm.go",
        "sum_arm.s",
        "sum_noasm.go",
        "sum_ref.go",
        "sum_s390x.go",
        "sum_s390x.s",
        "sum_vmsl_s390x.s",
    ],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/crypto/poly1305",
    importpath = "golang.org/x/crypto/poly1305",
    visibility = ["//visibility:public"],
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
