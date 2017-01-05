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
        "conversion.go",
        "defaults.go",
        "doc.go",
        "generated.pb.go",
        "helpers.go",
        "register.go",
        "types.generated.go",
        "types.go",
        "types_swagger_doc_generated.go",
        "zz_generated.conversion.go",
        "zz_generated.deepcopy.go",
        "zz_generated.defaults.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/meta/v1:go_default_library",
        "//pkg/apis/rbac:go_default_library",
        "//pkg/conversion:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/runtime/schema:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/watch/versioned:go_default_library",
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:github.com/ugorji/go/codec",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
    ],
)

go_test(
    name = "go_default_xtest",
    srcs = ["conversion_test.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/apis/rbac:go_default_library",
        "//pkg/apis/rbac/install:go_default_library",
        "//pkg/apis/rbac/v1alpha1:go_default_library",
    ],
)
