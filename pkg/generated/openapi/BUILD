package(default_visibility = ["//visibility:public"])

load("//build:code_generation.bzl", "gen_openapi", "openapi_deps")
load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

gen_openapi(
    outs = ["zz_generated.openapi.go"],
    exclude_pkgs = [
        "staging/src/k8s.io/code-generator",
        "staging/src/k8s.io/sample-apiserver",
    ],
    output_pkg = "k8s.io/kubernetes/pkg/generated/openapi",
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "zz_generated.openapi.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/generated/openapi",
    deps = openapi_deps(),  # keep
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

go_test(
    name = "go_default_test",
    srcs = ["openapi_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/util/diff:go_default_library",
        "//vendor/github.com/go-openapi/spec:go_default_library",
    ],
)
