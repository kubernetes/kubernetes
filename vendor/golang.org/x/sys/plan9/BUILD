load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "asm.s",
        "asm_plan9_386.s",
        "asm_plan9_amd64.s",
        "asm_plan9_arm.s",
        "const_plan9.go",
        "dir_plan9.go",
        "env_plan9.go",
        "errors_plan9.go",
        "pwd_go15_plan9.go",
        "pwd_plan9.go",
        "race.go",
        "race0.go",
        "str.go",
        "syscall.go",
        "syscall_plan9.go",
        "zsyscall_plan9_386.go",
        "zsyscall_plan9_amd64.go",
        "zsyscall_plan9_arm.go",
        "zsysnum_plan9.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/sys/plan9",
    importpath = "golang.org/x/sys/plan9",
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:plan9": [
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
