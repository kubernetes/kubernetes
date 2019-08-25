load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "confirmeddir.go",
        "fakefile.go",
        "fakefileinfo.go",
        "fakefs.go",
        "fs.go",
        "realfile.go",
        "realfs.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/kustomize/pkg/fs",
    importpath = "sigs.k8s.io/kustomize/pkg/fs",
    visibility = ["//visibility:public"],
    deps = ["//vendor/sigs.k8s.io/kustomize/pkg/pgmconfig:go_default_library"],
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

go_test(
    name = "go_default_test",
    srcs = [
        "confirmeddir_test.go",
        "fakefs_test.go",
        "realfs_test.go",
    ],
    embed = [":go_default_library"],
    tags = ["manual"],
)
