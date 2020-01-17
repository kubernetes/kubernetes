package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "framework.go",
        "kubectl.go",
        "portforward.go",
    ],
    importpath = "k8s.io/kubernetes/test/e2e/kubectl",
    deps = [
        "//pkg/controller:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/api/rbac/v1:go_default_library",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/meta:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1/unstructured:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/fields:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/net:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/rand:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/uuid:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/watch:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authentication/serviceaccount:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/registry/generic/registry:go_default_library",
        "//staging/src/k8s.io/client-go/discovery:go_default_library",
        "//staging/src/k8s.io/client-go/dynamic:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//staging/src/k8s.io/client-go/tools/watch:go_default_library",
        "//staging/src/k8s.io/kubectl/pkg/polymorphichelpers:go_default_library",
        "//test/e2e/common:go_default_library",
        "//test/e2e/framework:go_default_library",
        "//test/e2e/framework/auth:go_default_library",
        "//test/e2e/framework/endpoints:go_default_library",
        "//test/e2e/framework/job:go_default_library",
        "//test/e2e/framework/kubectl:go_default_library",
        "//test/e2e/framework/pod:go_default_library",
        "//test/e2e/framework/service:go_default_library",
        "//test/e2e/framework/skipper:go_default_library",
        "//test/e2e/framework/testfiles:go_default_library",
        "//test/e2e/scheduling:go_default_library",
        "//test/integration/etcd:go_default_library",
        "//test/utils:go_default_library",
        "//test/utils/crd:go_default_library",
        "//test/utils/image:go_default_library",
        "//vendor/github.com/elazarl/goproxy:go_default_library",
        "//vendor/github.com/googleapis/gnostic/OpenAPIv2:go_default_library",
        "//vendor/github.com/onsi/ginkgo:go_default_library",
        "//vendor/github.com/onsi/gomega:go_default_library",
        "//vendor/golang.org/x/net/websocket:go_default_library",
        "//vendor/k8s.io/utils/exec:go_default_library",
        "//vendor/k8s.io/utils/pointer:go_default_library",
        "//vendor/sigs.k8s.io/yaml:go_default_library",
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
