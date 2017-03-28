package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["portworx_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1:go_default_library",
        "//pkg/util/mount:go_default_library",
        "//pkg/volume:go_default_library",
        "//pkg/volume/testing:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/client-go/util/testing",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "portworx.go",
        "portworx_util.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1:go_default_library",
        "//pkg/util/exec:go_default_library",
        "//pkg/util/mount:go_default_library",
        "//pkg/util/strings:go_default_library",
        "//pkg/volume:go_default_library",
        "//pkg/volume/util:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/libopenstorage/openstorage/api",
        "//vendor:github.com/libopenstorage/openstorage/api/client",
        "//vendor:github.com/libopenstorage/openstorage/api/client/volume",
        "//vendor:github.com/libopenstorage/openstorage/api/spec",
        "//vendor:github.com/libopenstorage/openstorage/volume",
        "//vendor:k8s.io/apimachinery/pkg/api/resource",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/types",
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
