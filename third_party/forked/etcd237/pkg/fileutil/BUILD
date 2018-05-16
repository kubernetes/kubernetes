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
    embed = [":go_default_library"],
)

go_library(
    name = "go_default_library",
    srcs = [
        "fileutil.go",
        "lock.go",
        "lock_plan9.go",
        "lock_solaris.go",
        "lock_unix.go",
        "lock_windows.go",
        "perallocate_unsupported.go",
        "preallocate.go",
        "purge.go",
        "sync.go",
        "sync_linux.go",
    ],
    importpath = "k8s.io/kubernetes/third_party/forked/etcd237/pkg/fileutil",
    deps = [
        "//vendor/github.com/coreos/pkg/capnslog:go_default_library",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
)
