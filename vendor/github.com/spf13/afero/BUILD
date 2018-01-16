load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "afero.go",
        "basepath.go",
        "cacheOnReadFs.go",
        "copyOnWriteFs.go",
        "httpFs.go",
        "ioutil.go",
        "memmap.go",
        "memradix.go",
        "os.go",
        "path.go",
        "readonlyfs.go",
        "regexpfs.go",
        "sftp.go",
        "unionFile.go",
        "util.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "const_win_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "const_bsds.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "const_bsds.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "const_bsds.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "const_win_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "const_win_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "const_bsds.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "const_bsds.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "const_win_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "const_win_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "const_win_unix.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/spf13/afero",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/pkg/sftp:go_default_library",
        "//vendor/github.com/spf13/afero/mem:go_default_library",
        "//vendor/github.com/spf13/afero/sftp:go_default_library",
        "//vendor/golang.org/x/text/transform:go_default_library",
        "//vendor/golang.org/x/text/unicode/norm:go_default_library",
    ],
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
        "//vendor/github.com/spf13/afero/mem:all-srcs",
        "//vendor/github.com/spf13/afero/sftp:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
