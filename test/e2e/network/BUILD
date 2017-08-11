package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "dns.go",
        "dns_common.go",
        "dns_configmap.go",
        "doc.go",
        "firewall.go",
        "framework.go",
        "ingress.go",
        "kube_proxy.go",
        "network_policy.go",
        "networking.go",
        "networking_perf.go",
        "no_snat.go",
        "proxy.go",
        "service.go",
        "service_latency.go",
        "serviceloadbalancers.go",
        "util_iperf.go",
    ],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/testapi:go_default_library",
        "//pkg/apis/networking:go_default_library",
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//pkg/cloudprovider:go_default_library",
        "//pkg/cloudprovider/providers/gce:go_default_library",
        "//pkg/controller/endpoint:go_default_library",
        "//pkg/kubelet/apis:go_default_library",
        "//pkg/master/ports:go_default_library",
        "//test/e2e/framework:go_default_library",
        "//test/e2e/manifest:go_default_library",
        "//test/images/net/nat:go_default_library",
        "//test/utils:go_default_library",
        "//vendor/github.com/onsi/ginkgo:go_default_library",
        "//vendor/github.com/onsi/gomega:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/api/rbac/v1beta1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/fields:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/intstr:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/net:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/uuid:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/watch:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/authentication/serviceaccount:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes:go_default_library",
        "//vendor/k8s.io/client-go/rest:go_default_library",
        "//vendor/k8s.io/client-go/tools/cache:go_default_library",
        "//vendor/k8s.io/client-go/util/flowcontrol:go_default_library",
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
