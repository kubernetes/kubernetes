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
        "crd_test.go",
        "kube_apiserver_test.go",
        "main_test.go",
        "secrets_transformation_test.go",
        "synthetic_master_test.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "kms_transformation_test.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "kms_transformation_test.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "kms_transformation_test.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "kms_transformation_test.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "kms_transformation_test.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "kms_transformation_test.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "kms_transformation_test.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "kms_transformation_test.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "kms_transformation_test.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "kms_transformation_test.go",
        ],
        "//conditions:default": [],
    }),
    embed = [":go_default_library"],
    tags = ["integration"],
    deps = [
        "//cmd/kube-apiserver/app/testing:go_default_library",
        "//pkg/api/testapi:go_default_library",
        "//pkg/apis/core:go_default_library",
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//pkg/master:go_default_library",
        "//test/integration:go_default_library",
        "//test/integration/framework:go_default_library",
        "//vendor/github.com/ghodss/yaml:go_default_library",
        "//vendor/k8s.io/api/admissionregistration/v1alpha1:go_default_library",
        "//vendor/k8s.io/api/apps/v1:go_default_library",
        "//vendor/k8s.io/api/apps/v1beta1:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/api/networking/v1:go_default_library",
        "//vendor/k8s.io/apiextensions-apiserver/pkg/apis/apiextensions:go_default_library",
        "//vendor/k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1:go_default_library",
        "//vendor/k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1/unstructured:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/authentication/group:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/authentication/request/bearertoken:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/authentication/user:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/authorization/authorizer:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/authorization/authorizerfactory:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/features:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/registry/generic/registry:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/server/options/encryptionconfig:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/storage/value:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/storage/value/encrypt/aes:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/util/feature/testing:go_default_library",
        "//vendor/k8s.io/apiserver/plugin/pkg/authenticator/token/tokentest:go_default_library",
        "//vendor/k8s.io/client-go/dynamic:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/typed/core/v1:go_default_library",
        "//vendor/k8s.io/client-go/rest:go_default_library",
        "//vendor/k8s.io/kube-aggregator/pkg/apis/apiregistration:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "//vendor/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "//vendor/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "//vendor/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//vendor/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "//vendor/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "//vendor/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "//vendor/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "//vendor/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "//vendor/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
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
        "transformation_testcase.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "kms_plugin_mock.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "kms_plugin_mock.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "kms_plugin_mock.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "kms_plugin_mock.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "kms_plugin_mock.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "kms_plugin_mock.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "kms_plugin_mock.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "kms_plugin_mock.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "kms_plugin_mock.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "kms_plugin_mock.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "k8s.io/kubernetes/test/integration/master",
    deps = [
        "//cmd/kube-apiserver/app/testing:go_default_library",
        "//test/integration:go_default_library",
        "//test/integration/framework:go_default_library",
        "//vendor/github.com/coreos/etcd/clientv3:go_default_library",
        "//vendor/github.com/ghodss/yaml:go_default_library",
        "//vendor/github.com/prometheus/client_golang/prometheus:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/server/options/encryptionconfig:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/storage/storagebackend:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/storage/value:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "//vendor/github.com/golang/glog:go_default_library",
            "//vendor/golang.org/x/sys/unix:go_default_library",
            "//vendor/google.golang.org/grpc:go_default_library",
            "//vendor/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "//vendor/github.com/golang/glog:go_default_library",
            "//vendor/golang.org/x/sys/unix:go_default_library",
            "//vendor/google.golang.org/grpc:go_default_library",
            "//vendor/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "//vendor/github.com/golang/glog:go_default_library",
            "//vendor/golang.org/x/sys/unix:go_default_library",
            "//vendor/google.golang.org/grpc:go_default_library",
            "//vendor/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//vendor/github.com/golang/glog:go_default_library",
            "//vendor/golang.org/x/sys/unix:go_default_library",
            "//vendor/google.golang.org/grpc:go_default_library",
            "//vendor/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/github.com/golang/glog:go_default_library",
            "//vendor/golang.org/x/sys/unix:go_default_library",
            "//vendor/google.golang.org/grpc:go_default_library",
            "//vendor/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "//vendor/github.com/golang/glog:go_default_library",
            "//vendor/golang.org/x/sys/unix:go_default_library",
            "//vendor/google.golang.org/grpc:go_default_library",
            "//vendor/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "//vendor/github.com/golang/glog:go_default_library",
            "//vendor/golang.org/x/sys/unix:go_default_library",
            "//vendor/google.golang.org/grpc:go_default_library",
            "//vendor/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "//vendor/github.com/golang/glog:go_default_library",
            "//vendor/golang.org/x/sys/unix:go_default_library",
            "//vendor/google.golang.org/grpc:go_default_library",
            "//vendor/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "//vendor/github.com/golang/glog:go_default_library",
            "//vendor/golang.org/x/sys/unix:go_default_library",
            "//vendor/google.golang.org/grpc:go_default_library",
            "//vendor/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "//vendor/github.com/golang/glog:go_default_library",
            "//vendor/golang.org/x/sys/unix:go_default_library",
            "//vendor/google.golang.org/grpc:go_default_library",
            "//vendor/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
        ],
        "//conditions:default": [],
    }),
)
