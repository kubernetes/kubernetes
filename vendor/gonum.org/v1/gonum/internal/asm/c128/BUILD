load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "axpyinc_amd64.s",
        "axpyincto_amd64.s",
        "axpyunitary_amd64.s",
        "axpyunitaryto_amd64.s",
        "doc.go",
        "dotcinc_amd64.s",
        "dotcunitary_amd64.s",
        "dotuinc_amd64.s",
        "dotuunitary_amd64.s",
        "dscalinc_amd64.s",
        "dscalunitary_amd64.s",
        "scal.go",
        "scalUnitary_amd64.s",
        "scalinc_amd64.s",
        "stubs_amd64.go",
        "stubs_noasm.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/gonum.org/v1/gonum/internal/asm/c128",
    importpath = "gonum.org/v1/gonum/internal/asm/c128",
    visibility = ["//vendor/gonum.org/v1/gonum:__subpackages__"],
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
