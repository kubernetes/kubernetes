load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "seccomp.go",
        "seccomp_internal.go",
    ],
    cgo = True,
    clinkopts = select({"@io_bazel_rules_go//go/platform:linux":["-lseccomp",],"//conditions:default":[],}),
    importmap = "k8s.io/kubernetes/vendor/github.com/seccomp/libseccomp-golang",
    importpath = "github.com/seccomp/libseccomp-golang",
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
