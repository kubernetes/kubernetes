load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "backup.go",
        "callback.go",
        "doc.go",
        "error.go",
        "sqlite3.go",
        "sqlite3-binding.c",
        "sqlite3-binding.h",
        "sqlite3_context.go",
        "sqlite3_go18.go",
        "sqlite3_load_extension.go",
        "sqlite3_other.go",
        "sqlite3_type.go",
        "sqlite3_windows.go",
        "sqlite3ext.h",
    ],
    cgo = True,
    clinkopts = select({
        "@io_bazel_rules_go//go/platform:linux": [
            "-ldl",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "-lc",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "-lmingwex -lmingw32",
        ],
        "//conditions:default": [],
    }),
    copts = [
        "-DSQLITE_DISABLE_INTRINSIC",
        "-DSQLITE_ENABLE_FTS3 -DSQLITE_ENABLE_FTS3_PARENTHESIS -DSQLITE_ENABLE_FTS4_UNICODE61",
        "-DSQLITE_ENABLE_RTREE -DSQLITE_THREADSAFE=1",
        "-DSQLITE_TRACE_SIZE_LIMIT=15",
        "-Wno-deprecated-declarations",
        "-std=gnu99",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "-Ivendor/github.com/mattn/go-sqlite3",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "-Ivendor/github.com/mattn/go-sqlite3",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "-Ivendor/github.com/mattn/go-sqlite3",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "-Ivendor/github.com/mattn/go-sqlite3",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "-Ivendor/github.com/mattn/go-sqlite3",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "-Ivendor/github.com/mattn/go-sqlite3",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "-Ivendor/github.com/mattn/go-sqlite3",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "-Ivendor/github.com/mattn/go-sqlite3",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "-Ivendor/github.com/mattn/go-sqlite3",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "-Ivendor/github.com/mattn/go-sqlite3",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "-Ivendor/github.com/mattn/go-sqlite3 -fno-stack-check -fno-stack-protector -mno-stack-arg-probe",
        ],
        "//conditions:default": [],
    }) + select({
        "@io_bazel_rules_go//go/platform:windows_386": [
            "-D_USE_32BIT_TIME_T",
        ],
        "//conditions:default": [],
    }),
    importmap = "k8s.io/kubernetes/vendor/github.com/mattn/go-sqlite3",
    importpath = "github.com/mattn/go-sqlite3",
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
