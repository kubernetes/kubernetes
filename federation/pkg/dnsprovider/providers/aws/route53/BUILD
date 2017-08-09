package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "interface.go",
        "route53.go",
        "rrchangeset.go",
        "rrset.go",
        "rrsets.go",
        "zone.go",
        "zones.go",
    ],
    deps = [
        "//federation/pkg/dnsprovider:go_default_library",
        "//federation/pkg/dnsprovider/providers/aws/route53/stubs:go_default_library",
        "//federation/pkg/dnsprovider/rrstype:go_default_library",
        "//vendor/github.com/aws/aws-sdk-go/aws:go_default_library",
        "//vendor/github.com/aws/aws-sdk-go/aws/request:go_default_library",
        "//vendor/github.com/aws/aws-sdk-go/aws/session:go_default_library",
        "//vendor/github.com/aws/aws-sdk-go/service/route53:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/uuid:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["route53_test.go"],
    library = ":go_default_library",
    deps = [
        "//federation/pkg/dnsprovider:go_default_library",
        "//federation/pkg/dnsprovider/providers/aws/route53/stubs:go_default_library",
        "//federation/pkg/dnsprovider/rrstype:go_default_library",
        "//federation/pkg/dnsprovider/tests:go_default_library",
        "//vendor/github.com/aws/aws-sdk-go/aws:go_default_library",
        "//vendor/github.com/aws/aws-sdk-go/service/route53:go_default_library",
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
        "//federation/pkg/dnsprovider/providers/aws/route53/stubs:all-srcs",
    ],
    tags = ["automanaged"],
)
