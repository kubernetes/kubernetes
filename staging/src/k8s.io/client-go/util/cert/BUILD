package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = [
        "csr_test.go",
        "pem_test.go",
    ],
    data = glob(["testdata/**"]),
    embed = [":go_default_library"],
    importpath = "k8s.io/client-go/util/cert",
)

go_library(
    name = "go_default_library",
    srcs = [
        "cert.go",
        "csr.go",
        "io.go",
        "pem.go",
    ],
    data = [
        "testdata/dontUseThisKey.pem",
    ],
    importpath = "k8s.io/client-go/util/cert",
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
        "//staging/src/k8s.io/client-go/util/cert/triple:all-srcs",
    ],
    tags = ["automanaged"],
)
