package(default_visibility = ["//pkg/kubectl/visibility:kubectl_allowed_deps"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
)
load("//pkg/version:def.bzl", "version_x_defs")

go_binary(
    name = "kubectl",
    gc_linkopts = select({
        # Mac OS X doesn't support static binaries:
        # https://developer.apple.com/library/content/qa/qa1118/_index.html
        "@io_bazel_rules_go//go/platform:darwin_amd64": [],
        "//conditions:default": [
            "-linkmode",
            "external",
            "-extldflags",
            "-static",
        ],
    }),
    library = ":go_default_library",
    x_defs = version_x_defs(),
)

go_library(
    name = "go_default_library",
    srcs = ["kubectl.go"],
    deps = ["//cmd/kubectl/app:go_default_library"],
)

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
)

filegroup(
    name = "all-srcs",
    srcs = [
        ":package-srcs",
        "//cmd/kubectl/app:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
