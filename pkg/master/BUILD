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
        "controller.go",
        "doc.go",
        "import_known_versions.go",
        "master.go",
        "services.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//cmd/kube-apiserver/app/options:go_default_library",
        "//pkg/api/endpoints:go_default_library",
        "//pkg/apis/componentconfig/install:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/core/v1:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/core/internalversion:go_default_library",
        "//pkg/kubelet/client:go_default_library",
        "//pkg/master/thirdparty:go_default_library",
        "//pkg/master/tunneler:go_default_library",
        "//pkg/registry/apps/rest:go_default_library",
        "//pkg/registry/authentication/rest:go_default_library",
        "//pkg/registry/authorization/rest:go_default_library",
        "//pkg/registry/autoscaling/rest:go_default_library",
        "//pkg/registry/batch/rest:go_default_library",
        "//pkg/registry/certificates/rest:go_default_library",
        "//pkg/registry/core/rangeallocation:go_default_library",
        "//pkg/registry/core/rest:go_default_library",
        "//pkg/registry/core/service/ipallocator:go_default_library",
        "//pkg/registry/core/service/ipallocator/controller:go_default_library",
        "//pkg/registry/core/service/portallocator/controller:go_default_library",
        "//pkg/registry/extensions/rest:go_default_library",
        "//pkg/registry/policy/rest:go_default_library",
        "//pkg/registry/rbac/rest:go_default_library",
        "//pkg/registry/storage/rest:go_default_library",
        "//pkg/routes:go_default_library",
        "//pkg/util/async:go_default_library",
        "//pkg/util/node:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "controller_test.go",
        "import_known_versions_test.go",
        "master_openapi_test.go",
        "master_test.go",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api/testapi:go_default_library",
        "//pkg/client/clientset_generated/clientset/fake:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/fake:go_default_library",
        "//pkg/generated/openapi:go_default_library",
        "//pkg/kubelet/client:go_default_library",
        "//pkg/version:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//pkg/master/ports:all-srcs",
        "//pkg/master/thirdparty:all-srcs",
        "//pkg/master/tunneler:all-srcs",
    ],
    tags = ["automanaged"],
)
