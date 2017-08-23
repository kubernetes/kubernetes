package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = [
        "fileutil_test.go",
        "lock_test.go",
        "preallocate_test.go",
        "purge_test.go",
    ],
    library = ":go_default_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "fileutil.go",
        "lock.go",
        "lock_unix.go",
        "perallocate_unsupported.go",
        "purge.go",
        "sync.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "preallocate.go",
            "sync_linux.go",
        ],
        "@io_bazel_rules_go//go/platform:windows_amd64": [
            "lock_windows.go",
        ],
        "//conditions:default": [],
    }),
    deps = ["//vendor/github.com/coreos/pkg/capnslog:go_default_library"],
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
