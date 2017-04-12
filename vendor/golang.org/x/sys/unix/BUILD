package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "asm.s",
        "asm_linux_amd64.s",
        "bluetooth_linux.go",
        "constants.go",
        "env_unix.go",
        "env_unset.go",
        "flock.go",
        "race0.go",
        "sockcmsg_linux.go",
        "sockcmsg_unix.go",
        "str.go",
        "syscall.go",
        "syscall_linux.go",
        "syscall_linux_amd64.go",
        "syscall_unix.go",
        "zerrors_linux_amd64.go",
        "zsyscall_linux_amd64.go",
        "zsysnum_linux_amd64.go",
        "ztypes_linux_amd64.go",
    ],
    tags = ["automanaged"],
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
