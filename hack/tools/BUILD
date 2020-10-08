load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["tools.go"],
    importpath = "k8s.io/kubernetes/hack/tools",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/bazelbuild/buildtools/buildozer:go_default_library",
        "//vendor/github.com/cespare/prettybench:go_default_library",
        "//vendor/github.com/client9/misspell/cmd/misspell:go_default_library",
        "//vendor/golang.org/x/lint/golint:go_default_library",
        "//vendor/gotest.tools:go_default_library",
        "//vendor/gotest.tools/gotestsum:go_default_library",
        "//vendor/honnef.co/go/tools/cmd/staticcheck:go_default_library",
        "//vendor/k8s.io/repo-infra/cmd/kazel:go_default_library",
        "@bazel_gazelle//cmd/gazelle:go_default_library",
    ],
)
