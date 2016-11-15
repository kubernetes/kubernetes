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
        "configmap.go",
        "doc.go",
        "persistent_volume_claims.go",
        "pods.go",
        "registry.go",
        "replication_controllers.go",
        "resource_quotas.go",
        "secrets.go",
        "services.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/admission:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/api/resource:go_default_library",
        "//pkg/api/unversioned:go_default_library",
        "//pkg/api/validation:go_default_library",
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//pkg/controller/informers:go_default_library",
        "//pkg/kubelet/qos:go_default_library",
        "//pkg/quota:go_default_library",
        "//pkg/quota/generic:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/util/sets:go_default_library",
        "//pkg/util/validation/field:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "persistent_volume_claims_test.go",
        "pods_test.go",
        "services_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/resource:go_default_library",
        "//pkg/api/unversioned:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/fake:go_default_library",
        "//pkg/quota:go_default_library",
    ],
)
