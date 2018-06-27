load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "go_above_17.go",
        "go_above_19.go",
        "go_below_17.go",
        "go_below_19.go",
        "reflect2.go",
        "reflect2_kind.go",
        "relfect2_mips64x.s",
        "relfect2_mipsx.s",
        "relfect2_ppc64x.s",
        "safe_field.go",
        "safe_map.go",
        "safe_slice.go",
        "safe_struct.go",
        "safe_type.go",
        "type_map.go",
        "unsafe_array.go",
        "unsafe_eface.go",
        "unsafe_field.go",
        "unsafe_iface.go",
        "unsafe_link.go",
        "unsafe_map.go",
        "unsafe_ptr.go",
        "unsafe_slice.go",
        "unsafe_struct.go",
        "unsafe_type.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:darwin_amd64": [
            "reflect2_amd64.s",
        ],
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "reflect2_amd64.s",
        ],
        "@io_bazel_rules_go//go/platform:windows_amd64": [
            "reflect2_amd64.s",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/modern-go/reflect2",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/modern-go/concurrent:go_default_library"],
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
