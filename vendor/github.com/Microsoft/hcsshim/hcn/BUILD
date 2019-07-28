load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "hcn.go",
        "hcnendpoint.go",
        "hcnerrors.go",
        "hcnglobals.go",
        "hcnloadbalancer.go",
        "hcnnamespace.go",
        "hcnnetwork.go",
        "hcnpolicy.go",
        "hcnsupport.go",
        "zsyscall_windows.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/Microsoft/hcsshim/hcn",
    importpath = "github.com/Microsoft/hcsshim/hcn",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/Microsoft/hcsshim/internal/cni:go_default_library",
        "//vendor/github.com/Microsoft/hcsshim/internal/guid:go_default_library",
        "//vendor/github.com/Microsoft/hcsshim/internal/hcserror:go_default_library",
        "//vendor/github.com/Microsoft/hcsshim/internal/interop:go_default_library",
        "//vendor/github.com/Microsoft/hcsshim/internal/regstate:go_default_library",
        "//vendor/github.com/Microsoft/hcsshim/internal/runhcs:go_default_library",
        "//vendor/github.com/sirupsen/logrus:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:windows": [
            "//vendor/golang.org/x/sys/windows:go_default_library",
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
    visibility = ["//visibility:public"],
)
