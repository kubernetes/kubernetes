load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["main.go"],
    importpath = "k8s.io/kubernetes/cmd/staginghub",
    visibility = ["//visibility:private"],
    deps = [
        "//cmd/staginghub/override:go_default_library",
        "//third_party/go-git-http:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
    ],
)

go_binary(
    name = "staginghub",
    importpath = "k8s.io/kubernetes/cmd/staginghub",
    library = ":go_default_library",
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
    srcs = [
        ":package-srcs",
        "//cmd/staginghub/override:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
