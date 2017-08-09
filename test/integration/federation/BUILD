package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_test",
)

go_test(
    name = "go_default_test",
    size = "large",
    srcs = [
        "api_test.go",
        "crud_test.go",
        "main_test.go",
    ],
    tags = ["integration"],
    deps = [
        "//federation/apis/federation/v1beta1:go_default_library",
        "//federation/pkg/federatedtypes:go_default_library",
        "//federation/pkg/federatedtypes/crudtester:go_default_library",
        "//test/integration/federation/framework:go_default_library",
        "//test/integration/framework:go_default_library",
        "//vendor/github.com/pborman/uuid:go_default_library",
        "//vendor/github.com/stretchr/testify/assert:go_default_library",
        "//vendor/k8s.io/api/autoscaling/v1:go_default_library",
        "//vendor/k8s.io/api/batch/v1:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/api/extensions/v1beta1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
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
        "//test/integration/federation/framework:all-srcs",
    ],
    tags = ["automanaged"],
)
