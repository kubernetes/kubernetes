load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "cmdproto.go",
        "markdown.go",
        "normalizers.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/kubectl/cmd/util/cmdproto",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/kubectl/cmd/util:go_default_library",
        "//pkg/kubectl/cmd/util/cmdproto/k8s_io_kubectl_cmd:go_default_library",
        "//pkg/kubectl/util/i18n:go_default_library",
        "//vendor/github.com/MakeNowJust/heredoc:go_default_library",
        "//vendor/github.com/golang/protobuf/descriptor:go_default_library",
        "//vendor/github.com/golang/protobuf/proto:go_default_library",
        "//vendor/github.com/russross/blackfriday:go_default_library",
        "//vendor/github.com/spf13/cobra:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["cmdproto_test.go"],
    importpath = "k8s.io/kubernetes/pkg/kubectl/cmd/util/cmdproto",
    library = ":go_default_library",
    deps = [
        "//pkg/kubectl/cmd/util/cmdproto/k8s_io_kubectl_cmd:go_default_library",
        "//vendor/github.com/spf13/cobra:go_default_library",
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
        "//pkg/kubectl/cmd/util/cmdproto/k8s_io_kubectl_cmd:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)