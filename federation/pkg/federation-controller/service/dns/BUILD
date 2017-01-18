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
        "//federation/pkg/dnsprovider/providers/google/clouddns:go_default_library",
        "//federation/pkg/federation-controller/service/ingress:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
    ],
)

go_library(
    name = "go_default_library",
    srcs = ["dns.go"],
    tags = ["automanaged"],
    deps = [
        "//federation/apis/federation:go_default_library",
        "//federation/client/clientset_generated/federation_clientset:go_default_library",
        "//federation/pkg/dnsprovider:go_default_library",
        "//federation/pkg/dnsprovider/rrstype:go_default_library",
        "//federation/pkg/federation-controller/service/ingress:go_default_library",
        "//federation/pkg/federation-controller/util:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/client/cache:go_default_library",
        "//pkg/client/legacylisters:go_default_library",
        "//pkg/controller:go_default_library",
        "//pkg/util/workqueue:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/wait",
        "//vendor:k8s.io/apimachinery/pkg/watch",
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
