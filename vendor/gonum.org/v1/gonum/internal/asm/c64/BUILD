load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "axpyinc_amd64.s",
        "axpyincto_amd64.s",
        "axpyunitary_amd64.s",
        "axpyunitaryto_amd64.s",
        "conj.go",
        "doc.go",
        "dotcinc_amd64.s",
        "dotcunitary_amd64.s",
        "dotuinc_amd64.s",
        "dotuunitary_amd64.s",
        "scal.go",
        "stubs_amd64.go",
        "stubs_noasm.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/gonum.org/v1/gonum/internal/asm/c64",
    importpath = "gonum.org/v1/gonum/internal/asm/c64",
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
