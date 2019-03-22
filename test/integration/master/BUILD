package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    size = "large",
    srcs = [
        "audit_dynamic_test.go",
        "audit_test.go",
        "crd_test.go",
        "graceful_shutdown_test.go",
        "kms_transformation_test.go",
        "kube_apiserver_test.go",
        "main_test.go",
        "secrets_transformation_test.go",
        "synthetic_master_test.go",
    ],
    embed = [":go_default_library"],
    tags = ["integration"],
    deps = [
        "//cmd/kube-apiserver/app/options:go_default_library",
        "//cmd/kube-apiserver/app/testing:go_default_library",
        "//pkg/apis/core:go_default_library",
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//pkg/master:go_default_library",
        "//staging/src/k8s.io/api/apps/v1:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/api/networking/v1:go_default_library",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/apis/apiextensions:go_default_library",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1:go_default_library",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1/unstructured:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/apis/audit:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/apis/audit/v1:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/apis/audit/v1beta1:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/apis/config/v1:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authentication/group:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authentication/request/bearertoken:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authentication/user:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authorization/authorizer:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authorization/authorizerfactory:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/features:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/registry/generic/registry:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/storage/value:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/aes:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/feature/testing:go_default_library",
        "//staging/src/k8s.io/apiserver/plugin/pkg/authenticator/token/tokentest:go_default_library",
        "//staging/src/k8s.io/client-go/dynamic:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes/typed/core/v1:go_default_library",
        "//staging/src/k8s.io/client-go/rest:go_default_library",
        "//staging/src/k8s.io/kube-aggregator/pkg/apis/apiregistration:go_default_library",
        "//test/integration:go_default_library",
        "//test/integration/etcd:go_default_library",
        "//test/integration/framework:go_default_library",
        "//test/utils:go_default_library",
        "//vendor/github.com/evanphx/json-patch:go_default_library",
        "//vendor/github.com/go-openapi/spec:go_default_library",
        "//vendor/github.com/stretchr/testify/require:go_default_library",
        "//vendor/sigs.k8s.io/yaml:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
        ],
        "//conditions:default": [],
    }),
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

go_library(
    name = "go_default_library",
    srcs = [
        "kms_plugin_mock.go",
        "transformation_testcase.go",
    ],
    importpath = "k8s.io/kubernetes/test/integration/master",
    deps = [
        "//cmd/kube-apiserver/app/testing:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/apis/config/v1:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/storage/storagebackend:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/storage/value:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//test/integration:go_default_library",
        "//test/integration/framework:go_default_library",
        "//vendor/github.com/coreos/etcd/clientv3:go_default_library",
        "//vendor/github.com/prometheus/client_golang/prometheus:go_default_library",
        "//vendor/sigs.k8s.io/yaml:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
            "//vendor/google.golang.org/grpc:go_default_library",
            "//vendor/k8s.io/klog:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
            "//vendor/google.golang.org/grpc:go_default_library",
            "//vendor/k8s.io/klog:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
            "//vendor/google.golang.org/grpc:go_default_library",
            "//vendor/k8s.io/klog:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
            "//vendor/google.golang.org/grpc:go_default_library",
            "//vendor/k8s.io/klog:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
            "//vendor/google.golang.org/grpc:go_default_library",
            "//vendor/k8s.io/klog:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
            "//vendor/google.golang.org/grpc:go_default_library",
            "//vendor/k8s.io/klog:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
            "//vendor/google.golang.org/grpc:go_default_library",
            "//vendor/k8s.io/klog:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
            "//vendor/google.golang.org/grpc:go_default_library",
            "//vendor/k8s.io/klog:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
            "//vendor/google.golang.org/grpc:go_default_library",
            "//vendor/k8s.io/klog:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
            "//vendor/google.golang.org/grpc:go_default_library",
            "//vendor/k8s.io/klog:go_default_library",
        ],
        "//conditions:default": [],
    }),
)
