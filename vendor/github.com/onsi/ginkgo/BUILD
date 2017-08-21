load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["ginkgo_dsl.go"],
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/onsi/ginkgo/config:go_default_library",
        "//vendor/github.com/onsi/ginkgo/internal/codelocation:go_default_library",
        "//vendor/github.com/onsi/ginkgo/internal/failer:go_default_library",
        "//vendor/github.com/onsi/ginkgo/internal/remote:go_default_library",
        "//vendor/github.com/onsi/ginkgo/internal/suite:go_default_library",
        "//vendor/github.com/onsi/ginkgo/internal/testingtproxy:go_default_library",
        "//vendor/github.com/onsi/ginkgo/internal/writer:go_default_library",
        "//vendor/github.com/onsi/ginkgo/reporters:go_default_library",
        "//vendor/github.com/onsi/ginkgo/reporters/stenographer:go_default_library",
        "//vendor/github.com/onsi/ginkgo/types:go_default_library",
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
        "//vendor/github.com/onsi/ginkgo/config:all-srcs",
        "//vendor/github.com/onsi/ginkgo/ginkgo:all-srcs",
        "//vendor/github.com/onsi/ginkgo/internal/codelocation:all-srcs",
        "//vendor/github.com/onsi/ginkgo/internal/containernode:all-srcs",
        "//vendor/github.com/onsi/ginkgo/internal/failer:all-srcs",
        "//vendor/github.com/onsi/ginkgo/internal/leafnodes:all-srcs",
        "//vendor/github.com/onsi/ginkgo/internal/remote:all-srcs",
        "//vendor/github.com/onsi/ginkgo/internal/spec:all-srcs",
        "//vendor/github.com/onsi/ginkgo/internal/spec_iterator:all-srcs",
        "//vendor/github.com/onsi/ginkgo/internal/specrunner:all-srcs",
        "//vendor/github.com/onsi/ginkgo/internal/suite:all-srcs",
        "//vendor/github.com/onsi/ginkgo/internal/testingtproxy:all-srcs",
        "//vendor/github.com/onsi/ginkgo/internal/writer:all-srcs",
        "//vendor/github.com/onsi/ginkgo/reporters:all-srcs",
        "//vendor/github.com/onsi/ginkgo/types:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
