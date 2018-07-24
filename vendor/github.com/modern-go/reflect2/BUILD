load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "go_above_17.go",
        "go_above_19.go",
        "go_below_17.go",
        "go_below_19.go",
        "reflect2.go",
        "reflect2_amd64.s",
        "reflect2_kind.go",
        "relfect2_386.s",
        "relfect2_amd64p32.s",
        "relfect2_arm.s",
        "relfect2_arm64.s",
        "relfect2_mips64x.s",
        "relfect2_mipsx.s",
        "relfect2_ppc64x.s",
        "relfect2_s390x.s",
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
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/modern-go/reflect2",
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
