package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "dns.go",
        "doc.go",
        "plugins.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//federation/pkg/dnsprovider/rrstype:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["dns_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = ["//federation/pkg/dnsprovider/rrstype:go_default_library"],
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
        "//federation/pkg/dnsprovider/providers/aws/route53:all-srcs",
        "//federation/pkg/dnsprovider/providers/coredns:all-srcs",
        "//federation/pkg/dnsprovider/providers/google/clouddns:all-srcs",
        "//federation/pkg/dnsprovider/rrstype:all-srcs",
        "//federation/pkg/dnsprovider/tests:all-srcs",
    ],
    tags = ["automanaged"],
)
