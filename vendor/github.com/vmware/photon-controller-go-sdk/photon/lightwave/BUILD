load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "jwttoken.go",
        "oidcclient.go",
        "oidcclient_sspi.go",
        "oidcclient_sspi_unsupported.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/vmware/photon-controller-go-sdk/photon/lightwave",
    importpath = "github.com/vmware/photon-controller-go-sdk/photon/lightwave",
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:windows": [
            "//vendor/github.com/vmware/photon-controller-go-sdk/SSPI:go_default_library",
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
