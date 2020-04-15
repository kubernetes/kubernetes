package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)
load("//build:platforms.bzl", "go_platform_constraint")

go_library(
    name = "go_default_library",
    srcs = [
        "etcd.go",
        "master_utils.go",
        "perf_utils.go",
        "serializer.go",
        "test_server.go",
        "util.go",
    ],
    data = select({
        go_platform_constraint(
            arch = "arm64",
            os = "linux",
        ): [
            "@com_coreos_etcd_arm64//:etcd",
        ],
        go_platform_constraint(
            arch = "ppc64le",
            os = "linux",
        ): [
            "@com_coreos_etcd_ppc64le//:etcd",
        ],
        "//conditions:default": [
            "@com_coreos_etcd_amd64//:etcd",
        ],
    }),
    importpath = "k8s.io/kubernetes/test/integration/framework",
    deps = [
        "//cmd/kube-apiserver/app:go_default_library",
        "//cmd/kube-apiserver/app/options:go_default_library",
        "//pkg/api/legacyscheme:go_default_library",
        "//pkg/apis/core/v1/helper:go_default_library",
        "//pkg/controller/nodelifecycle:go_default_library",
        "//pkg/generated/openapi:go_default_library",
        "//pkg/kubeapiserver:go_default_library",
        "//pkg/kubelet/client:go_default_library",
        "//pkg/master:go_default_library",
        "//pkg/util/env:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/fields:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/serializer/versioning:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authentication/authenticator:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authentication/authenticatorfactory:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authentication/request/union:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authentication/user:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authorization/authorizer:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authorization/authorizerfactory:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authorization/union:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/endpoints/openapi:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/server:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/server/options:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/server/storage:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/storage/storagebackend:go_default_library",
        "//staging/src/k8s.io/client-go/informers:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//staging/src/k8s.io/client-go/rest:go_default_library",
        "//staging/src/k8s.io/client-go/util/cert:go_default_library",
        "//staging/src/k8s.io/component-base/version:go_default_library",
        "//test/utils:go_default_library",
        "//vendor/github.com/go-openapi/spec:go_default_library",
        "//vendor/github.com/google/uuid:go_default_library",
        "//vendor/go.etcd.io/etcd/clientv3:go_default_library",
        "//vendor/google.golang.org/grpc/grpclog:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
        "//vendor/k8s.io/kube-openapi/pkg/common:go_default_library",
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
