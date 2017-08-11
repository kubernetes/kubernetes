package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
)

go_binary(
    name = "genfeddocs",
    library = ":go_default_library",
)

go_library(
    name = "go_default_library",
    srcs = ["gen_fed_docs.go"],
    deps = [
        "//cmd/genutils:go_default_library",
        "//federation/cmd/federation-apiserver/app:go_default_library",
        "//federation/cmd/federation-controller-manager/app:go_default_library",
        "//federation/cmd/kubefed/app:go_default_library",
        "//federation/pkg/kubefed:go_default_library",
        "//pkg/kubectl/cmd/util:go_default_library",
        "//vendor/github.com/spf13/cobra/doc:go_default_library",
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
