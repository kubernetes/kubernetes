load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "asm_aix_ppc64.s",
        "byteorder.go",
        "cpu.go",
        "cpu_arm.go",
        "cpu_gc_s390x.go",
        "cpu_gc_x86.go",
        "cpu_linux.go",
        "cpu_linux_arm64.go",
        "cpu_linux_ppc64x.go",
        "cpu_linux_s390x.go",
        "cpu_mips64x.go",
        "cpu_mipsx.go",
        "cpu_other_arm64.go",
        "cpu_s390x.s",
        "cpu_x86.go",
        "cpu_x86.s",
    ],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/sys/cpu",
    importpath = "golang.org/x/sys/cpu",
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
