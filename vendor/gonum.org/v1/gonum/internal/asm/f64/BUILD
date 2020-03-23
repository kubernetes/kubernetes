load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "abssum_amd64.s",
        "abssuminc_amd64.s",
        "add_amd64.s",
        "addconst_amd64.s",
        "axpy.go",
        "axpyinc_amd64.s",
        "axpyincto_amd64.s",
        "axpyunitary_amd64.s",
        "axpyunitaryto_amd64.s",
        "cumprod_amd64.s",
        "cumsum_amd64.s",
        "div_amd64.s",
        "divto_amd64.s",
        "doc.go",
        "dot.go",
        "dot_amd64.s",
        "ge_amd64.go",
        "ge_noasm.go",
        "gemvN_amd64.s",
        "gemvT_amd64.s",
        "ger_amd64.s",
        "l1norm_amd64.s",
        "linfnorm_amd64.s",
        "scal.go",
        "scalinc_amd64.s",
        "scalincto_amd64.s",
        "scalunitary_amd64.s",
        "scalunitaryto_amd64.s",
        "stubs_amd64.go",
        "stubs_noasm.go",
        "sum_amd64.s",
    ],
    importmap = "k8s.io/kubernetes/vendor/gonum.org/v1/gonum/internal/asm/f64",
    importpath = "gonum.org/v1/gonum/internal/asm/f64",
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
