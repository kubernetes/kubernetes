load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["fuzzer.go"],
    visibility = ["//visibility:public"],
    deps = [
<<<<<<< HEAD:staging/src/k8s.io/apiserver/pkg/apis/audit/fuzzer/BUILD
        "//vendor/github.com/google/gofuzz:go_default_library",
=======
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
>>>>>>> run hack/update-bazel.sh:staging/src/k8s.io/sample-controller/pkg/client/informers/externalversions/internalinterfaces/BUILD
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/serializer:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/apis/audit:go_default_library",
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
