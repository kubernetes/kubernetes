package(default_visibility = ["//visibility:public"])

load(
    "//build:go.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "probe.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/probe",
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
        "//pkg/probe/exec:all-srcs",
        "//pkg/probe/http:all-srcs",
        "//pkg/probe/tcp:all-srcs",
    ],
    tags = ["automanaged"],
)
