load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "aliases.go",
        "dll_windows.go",
        "empty.s",
        "env_windows.go",
        "eventlog.go",
        "exec_windows.go",
        "memory_windows.go",
        "race.go",
        "race0.go",
        "security_windows.go",
        "service.go",
        "str.go",
        "syscall.go",
        "syscall_windows.go",
        "types_windows.go",
        "types_windows_386.go",
        "types_windows_amd64.go",
        "types_windows_arm.go",
        "zerrors_windows.go",
        "zknownfolderids_windows.go",
        "zsyscall_windows.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/sys/windows",
    importpath = "golang.org/x/sys/windows",
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:windows": [
            "//vendor/golang.org/x/sys/internal/unsafeheader:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//vendor/golang.org/x/sys/windows/registry:all-srcs",
        "//vendor/golang.org/x/sys/windows/svc:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
