load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "mksyscall.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:windows_amd64": [
            "asm_windows_amd64.s",
            "dll_windows.go",
            "env_unset.go",
            "env_windows.go",
            "eventlog.go",
            "exec_windows.go",
            "memory_windows.go",
            "race0.go",
            "security_windows.go",
            "service.go",
            "str.go",
            "syscall.go",
            "syscall_windows.go",
            "zsyscall_windows.go",
            "ztypes_windows.go",
            "ztypes_windows_amd64.go",
        ],
        "//conditions:default": [],
    }),
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
