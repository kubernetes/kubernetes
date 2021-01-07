load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "csidriver.go",
        "csidriverspec.go",
        "csinode.go",
        "csinodedriver.go",
        "csinodespec.go",
        "storageclass.go",
        "tokenrequest.go",
        "volumeattachment.go",
        "volumeattachmentsource.go",
        "volumeattachmentspec.go",
        "volumeattachmentstatus.go",
        "volumeerror.go",
        "volumenoderesources.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/client-go/applyconfigurations/storage/v1beta1",
    importpath = "k8s.io/client-go/applyconfigurations/storage/v1beta1",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/api/storage/v1beta1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/client-go/applyconfigurations/core/v1:go_default_library",
        "//staging/src/k8s.io/client-go/applyconfigurations/meta/v1:go_default_library",
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
    visibility = ["//visibility:public"],
)
