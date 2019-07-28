package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["service.go"],
    importpath = "k8s.io/kubernetes/pkg/windows/service",
    deps = select({
        "@io_bazel_rules_go//go/platform:windows": [
            "//staging/src/k8s.io/apiserver/pkg/server:go_default_library",
            "//vendor/golang.org/x/sys/windows:go_default_library",
            "//vendor/golang.org/x/sys/windows/svc:go_default_library",
            "//vendor/k8s.io/klog:go_default_library",
        ],
        "//conditions:default": [],
    }),
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
