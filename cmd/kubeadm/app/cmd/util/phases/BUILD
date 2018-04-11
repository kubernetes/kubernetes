load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "builder.go",
        "doc.go",
        "phases.go",
    ],
    importpath = "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util/phases",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/spf13/cobra:go_default_library"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "builder_test.go",
        "phases_test.go",
    ],
    embed = [":go_default_library"],
    deps = ["//vendor/github.com/spf13/cobra:go_default_library"],
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
