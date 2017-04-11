package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["apiserver_test.go"],
    tags = ["automanaged"],
    deps = [
        "//cmd/kube-apiserver/app:go_default_library",
        "//cmd/kube-apiserver/app/options:go_default_library",
        "//test/integration/framework:go_default_library",
        "//vendor:github.com/stretchr/testify/assert",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/util/wait",
        "//vendor:k8s.io/client-go/kubernetes",
        "//vendor:k8s.io/client-go/rest",
        "//vendor:k8s.io/client-go/tools/clientcmd",
        "//vendor:k8s.io/client-go/tools/clientcmd/api",
        "//vendor:k8s.io/client-go/util/cert",
        "//vendor:k8s.io/kube-aggregator/pkg/apis/apiregistration/v1alpha1",
        "//vendor:k8s.io/kube-aggregator/pkg/client/clientset_generated/clientset",
        "//vendor:k8s.io/kube-aggregator/pkg/cmd/server",
        "//vendor:k8s.io/sample-apiserver/pkg/apis/wardle/v1alpha1",
        "//vendor:k8s.io/sample-apiserver/pkg/cmd/server",
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
