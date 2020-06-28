package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)
load("//build:bindata.bzl", "go_bindata")

go_library(
    name = "go_default_library",
    srcs = [
        "bindata.go",
        "gobindata_util.go",
        "main.go",
    ],
    importpath = "k8s.io/kubernetes/test/e2e/generated",
    deps = [
        "//vendor/k8s.io/klog/v2:go_default_library",
    ],
)

# IMPORTANT: if you make any changes here, you must also update hack/generate-bindata.sh.
go_bindata(
    name = "bindata",
    srcs = [
        "//test/conformance/testdata:all-srcs",
        "//test/e2e/testing-manifests:all-srcs",
        "//test/e2e_node/testing-manifests:all-srcs",
        "//test/fixtures:all-srcs",
        "//test/images:all-srcs",
    ],
    outs = ["bindata.go"],
    compress = True,
    include_metadata = False,
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
