package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["dns_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//federation/apis/federation/v1beta1:go_default_library",
        "//federation/pkg/dnsprovider/providers/google/clouddns:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/util/sets:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = ["dns.go"],
    tags = ["automanaged"],
    deps = [
        "//federation/pkg/dnsprovider:go_default_library",
        "//federation/pkg/dnsprovider/rrstype:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)
