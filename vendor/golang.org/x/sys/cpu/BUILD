load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "asm_aix_ppc64.s",
        "byteorder.go",
        "cpu.go",
        "cpu_aix.go",
        "cpu_arm.go",
        "cpu_arm64.go",
        "cpu_arm64.s",
        "cpu_gc_arm64.go",
        "cpu_gc_s390x.go",
        "cpu_gc_x86.go",
        "cpu_linux.go",
        "cpu_linux_arm.go",
        "cpu_linux_arm64.go",
        "cpu_linux_mips64x.go",
        "cpu_linux_noinit.go",
        "cpu_linux_ppc64x.go",
        "cpu_linux_s390x.go",
        "cpu_mips64x.go",
        "cpu_mipsx.go",
        "cpu_other_arm64.go",
        "cpu_riscv64.go",
        "cpu_s390x.s",
        "cpu_wasm.go",
        "cpu_x86.go",
        "cpu_x86.s",
        "hwcap_linux.go",
        "syscall_aix_ppc64_gc.go",
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
