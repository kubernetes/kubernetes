package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "cluster_helper.go",
        "dns.go",
        "doc.go",
        "endpoint_helper.go",
        "service_helper.go",
        "servicecontroller.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//federation/apis/federation/v1beta1:go_default_library",
        "//federation/client/cache:go_default_library",
        "//federation/client/clientset_generated/federation_release_1_5:go_default_library",
        "//federation/pkg/dnsprovider:go_default_library",
        "//federation/pkg/dnsprovider/rrstype:go_default_library",
        "//federation/pkg/federation-controller/util:go_default_library",
        "//pkg/api/errors:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/client/cache:go_default_library",
        "//pkg/client/clientset_generated/release_1_5:go_default_library",
        "//pkg/client/record:go_default_library",
        "//pkg/client/restclient:go_default_library",
        "//pkg/controller:go_default_library",
        "//pkg/conversion:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/util/runtime:go_default_library",
        "//pkg/util/sets:go_default_library",
        "//pkg/util/wait:go_default_library",
        "//pkg/util/workqueue:go_default_library",
        "//pkg/watch:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "dns_test.go",
        "endpoint_helper_test.go",
        "service_helper_test.go",
        "servicecontroller_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//federation/apis/federation/v1beta1:go_default_library",
        "//federation/pkg/dnsprovider/providers/google/clouddns:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/util/sets:go_default_library",
    ],
)
