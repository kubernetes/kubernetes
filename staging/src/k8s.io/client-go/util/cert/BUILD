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
    library = ":go_default_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "cert.go",
        "csr.go",
        "entropy.go",
        "entropy_other.go",
        "io.go",
        "pem.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "entropy_linux.go",
        ],
        "//conditions:default": [],
    }),
    data = [
        "testdata/dontUseThisKey.pem",
    ],
    deps = ["//vendor/github.com/golang/glog:go_default_library"],
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
