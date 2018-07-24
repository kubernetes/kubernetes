load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "bolt_386.go",
        "bolt_amd64.go",
        "bolt_arm.go",
        "bolt_arm64.go",
        "bolt_linux.go",
        "bolt_mips64x.go",
        "bolt_mipsx.go",
        "bolt_openbsd.go",
        "bolt_ppc64.go",
        "bolt_ppc64le.go",
        "bolt_s390x.go",
        "bolt_unix.go",
        "bolt_unix_solaris.go",
        "bolt_windows.go",
        "boltsync_unix.go",
        "bucket.go",
        "cursor.go",
        "db.go",
        "doc.go",
        "errors.go",
        "freelist.go",
        "node.go",
        "page.go",
        "tx.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/coreos/bbolt",
    importpath = "github.com/coreos/bbolt",
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:solaris": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
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
