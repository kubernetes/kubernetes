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
        "doc.go",
        "generated_expansion.go",
        "testgroup_client.go",
        "testtype.go",
        "testtype_expansion.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//cmd/libs/go2idl/client-gen/test_apis/testgroup.k8s.io:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/apimachinery/registered:go_default_library",
        "//pkg/client/restclient:go_default_library",
        "//pkg/watch:go_default_library",
    ],
)

go_test(
    name = "go_default_xtest",
    srcs = ["testgroup_test.go"],
    tags = ["automanaged"],
    deps = [
        "//cmd/libs/go2idl/client-gen/test_apis/testgroup.k8s.io:go_default_library",
        "//cmd/libs/go2idl/client-gen/test_apis/testgroup.k8s.io/install:go_default_library",
        "//cmd/libs/go2idl/client-gen/testoutput/clientset_generated/test_internalclientset/typed/testgroup.k8s.io/unversioned:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/api/testapi:go_default_library",
        "//pkg/api/unversioned:go_default_library",
        "//pkg/apimachinery/registered:go_default_library",
        "//pkg/client/restclient:go_default_library",
        "//pkg/client/unversioned/testclient/simple:go_default_library",
        "//pkg/labels:go_default_library",
    ],
)
