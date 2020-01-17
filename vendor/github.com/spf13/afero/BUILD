load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "afero.go",
        "basepath.go",
        "cacheOnReadFs.go",
        "const_bsds.go",
        "const_win_unix.go",
        "copyOnWriteFs.go",
        "httpFs.go",
        "ioutil.go",
        "lstater.go",
        "match.go",
        "memmap.go",
        "os.go",
        "path.go",
        "readonlyfs.go",
        "regexpfs.go",
        "unionFile.go",
        "util.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/spf13/afero",
    importpath = "github.com/spf13/afero",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/spf13/afero/mem:go_default_library",
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
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
