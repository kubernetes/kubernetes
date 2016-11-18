package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["rbac_test.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/testapi:go_default_library",
        "//pkg/apimachinery/registered:go_default_library",
        "//pkg/apis/rbac:go_default_library",
        "//pkg/auth/authenticator:go_default_library",
        "//pkg/auth/authenticator/bearertoken:go_default_library",
        "//pkg/auth/authorizer:go_default_library",
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//pkg/client/restclient:go_default_library",
        "//pkg/client/transport:go_default_library",
        "//pkg/master:go_default_library",
        "//pkg/registry/generic:go_default_library",
        "//pkg/registry/rbac/clusterrole:go_default_library",
        "//pkg/registry/rbac/clusterrole/storage:go_default_library",
        "//pkg/registry/rbac/clusterrolebinding:go_default_library",
        "//pkg/registry/rbac/clusterrolebinding/storage:go_default_library",
        "//pkg/registry/rbac/role:go_default_library",
        "//pkg/registry/rbac/role/storage:go_default_library",
        "//pkg/registry/rbac/rolebinding:go_default_library",
        "//pkg/registry/rbac/rolebinding/storage:go_default_library",
        "//pkg/runtime/schema:go_default_library",
        "//pkg/watch:go_default_library",
        "//plugin/pkg/auth/authenticator/token/anytoken:go_default_library",
        "//plugin/pkg/auth/authorizer/rbac:go_default_library",
        "//test/integration/framework:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)
