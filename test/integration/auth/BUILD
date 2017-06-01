package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = [
        "accessreview_test.go",
        "auth_test.go",
        "node_test.go",
        "rbac_test.go",
    ],
    tags = [
        "automanaged",
        "integration",
    ],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/testapi:go_default_library",
        "//pkg/apis/authentication/v1beta1:go_default_library",
        "//pkg/apis/authorization:go_default_library",
        "//pkg/apis/autoscaling:go_default_library",
        "//pkg/apis/extensions:go_default_library",
        "//pkg/apis/rbac:go_default_library",
        "//pkg/auth/authorizer/abac:go_default_library",
        "//pkg/auth/nodeidentifier:go_default_library",
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//pkg/client/informers/informers_generated/internalversion:go_default_library",
        "//pkg/kubeapiserver/authorizer:go_default_library",
        "//pkg/master:go_default_library",
        "//pkg/registry/rbac/clusterrole:go_default_library",
        "//pkg/registry/rbac/clusterrole/storage:go_default_library",
        "//pkg/registry/rbac/clusterrolebinding:go_default_library",
        "//pkg/registry/rbac/clusterrolebinding/storage:go_default_library",
        "//pkg/registry/rbac/role:go_default_library",
        "//pkg/registry/rbac/role/storage:go_default_library",
        "//pkg/registry/rbac/rolebinding:go_default_library",
        "//pkg/registry/rbac/rolebinding/storage:go_default_library",
        "//plugin/pkg/admission/admit:go_default_library",
        "//plugin/pkg/admission/noderestriction:go_default_library",
        "//plugin/pkg/auth/authorizer/rbac:go_default_library",
        "//plugin/pkg/auth/authorizer/rbac/bootstrappolicy:go_default_library",
        "//test/integration:go_default_library",
        "//test/integration/framework:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/watch:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/authentication/authenticator:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/authentication/group:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/authentication/request/bearertoken:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/authentication/serviceaccount:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/authentication/user:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/authorization/authorizer:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/authorization/authorizerfactory:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/registry/generic:go_default_library",
        "//vendor/k8s.io/apiserver/plugin/pkg/authenticator/token/anytoken:go_default_library",
        "//vendor/k8s.io/apiserver/plugin/pkg/authenticator/token/tokentest:go_default_library",
        "//vendor/k8s.io/apiserver/plugin/pkg/authenticator/token/webhook:go_default_library",
        "//vendor/k8s.io/client-go/rest:go_default_library",
        "//vendor/k8s.io/client-go/tools/clientcmd/api/v1:go_default_library",
        "//vendor/k8s.io/client-go/transport:go_default_library",
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
