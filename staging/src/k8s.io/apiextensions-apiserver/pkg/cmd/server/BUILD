package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["server.go"],
    data = glob(["testdata/**"]),
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apiextensions-apiserver/pkg/cmd/server",
    importpath = "k8s.io/apiextensions-apiserver/pkg/cmd/server",
    deps = [
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/cmd/server/options:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/server:go_default_library",
        "//vendor/github.com/spf13/cobra:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/cmd/server/options:all-srcs",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/cmd/server/testing:all-srcs",
    ],
    tags = ["automanaged"],
)
