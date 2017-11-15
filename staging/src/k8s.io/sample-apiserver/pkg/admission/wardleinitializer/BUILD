load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "interfaces.go",
        "wardleinitializer.go",
    ],
    importpath = "k8s.io/sample-apiserver/pkg/admission/wardleinitializer",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/client/informers/internalversion:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/admission:go_default_library",
    ],
)

go_test(
    name = "go_default_xtest",
    srcs = ["wardleinitializer_test.go"],
    importpath = "k8s.io/sample-apiserver/pkg/admission/wardleinitializer_test",
    deps = [
        ":go_default_library",
        "//pkg/client/clientset/internalversion/fake:go_default_library",
        "//pkg/client/informers/internalversion:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/admission:go_default_library",
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
