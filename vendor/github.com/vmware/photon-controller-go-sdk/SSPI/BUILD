load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = select({
        "@io_bazel_rules_go//go/platform:darwin_amd64": [
            "sspi_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "sspi_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:windows_amd64": [
            "sspi.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/vmware/photon-controller-go-sdk/SSPI",
    visibility = ["//visibility:public"],
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
