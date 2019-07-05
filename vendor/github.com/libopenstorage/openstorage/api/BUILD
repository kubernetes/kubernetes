load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "api.go",
        "api.pb.go",
        "status.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/libopenstorage/openstorage/api",
    importpath = "github.com/libopenstorage/openstorage/api",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/golang/protobuf/proto:go_default_library",
        "//vendor/github.com/golang/protobuf/ptypes/timestamp:go_default_library",
        "//vendor/github.com/mohae/deepcopy:go_default_library",
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
        "//vendor/github.com/libopenstorage/openstorage/api/client:all-srcs",
        "//vendor/github.com/libopenstorage/openstorage/api/spec:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
