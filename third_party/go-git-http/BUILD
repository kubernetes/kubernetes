load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "errors.go",
        "events.go",
        "git_reader.go",
        "githttp.go",
        "pktparser.go",
        "routing.go",
        "rpc_reader.go",
        "utils.go",
        "version.go",
    ],
    importpath = "k8s.io/kubernetes/third_party/go-git-http",
    visibility = ["//visibility:public"],
)

go_test(
    name = "go_default_test",
    srcs = ["pktparser_test.go"],
    data = glob(["testdata/**"]),
    importpath = "k8s.io/kubernetes/third_party/go-git-http",
    library = ":go_default_library",
)

go_test(
    name = "go_default_xtest",
    srcs = ["rpc_reader_test.go"],
    data = glob(["testdata/**"]),
    importpath = "k8s.io/kubernetes/third_party/go-git-http_test",
    deps = ["//vendor/github.com/AaronO/go-git-http:go_default_library"],
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
        "//third_party/go-git-http/auth:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
