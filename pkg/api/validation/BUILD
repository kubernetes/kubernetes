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
        "doc.go",
        "events.go",
        "schema.go",
        "validation.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/pod:go_default_library",
        "//pkg/api/service:go_default_library",
        "//pkg/api/util:go_default_library",
        "//pkg/apis/storage/util:go_default_library",
        "//pkg/capabilities:go_default_library",
        "//pkg/features:go_default_library",
        "//pkg/security/apparmor:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "events_test.go",
        "schema_test.go",
        "validation_test.go",
    ],
    data = [
        "testdata/v1/invalidPod.yaml",
        "testdata/v1/invalidPod1.json",
        "testdata/v1/invalidPod2.json",
        "testdata/v1/invalidPod3.json",
        "testdata/v1/invalidPod4.yaml",
        "testdata/v1/validPod.yaml",
        "//api/swagger-spec",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api/service:go_default_library",
        "//pkg/api/testapi:go_default_library",
        "//pkg/api/testing:go_default_library",
        "//pkg/apis/storage/util:go_default_library",
        "//pkg/capabilities:go_default_library",
        "//pkg/security/apparmor:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
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
