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
        "controllermanager.go",
        "plugins.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//federation/client/clientset_generated/federation_clientset:go_default_library",
        "//federation/cmd/federation-controller-manager/app/options:go_default_library",
        "//federation/pkg/dnsprovider:go_default_library",
        "//federation/pkg/dnsprovider/providers/aws/route53:go_default_library",
        "//federation/pkg/dnsprovider/providers/coredns:go_default_library",
        "//federation/pkg/dnsprovider/providers/google/clouddns:go_default_library",
        "//federation/pkg/federation-controller/cluster:go_default_library",
        "//federation/pkg/federation-controller/configmap:go_default_library",
        "//federation/pkg/federation-controller/daemonset:go_default_library",
        "//federation/pkg/federation-controller/deployment:go_default_library",
        "//federation/pkg/federation-controller/ingress:go_default_library",
        "//federation/pkg/federation-controller/namespace:go_default_library",
        "//federation/pkg/federation-controller/replicaset:go_default_library",
        "//federation/pkg/federation-controller/secret:go_default_library",
        "//federation/pkg/federation-controller/service:go_default_library",
        "//federation/pkg/federation-controller/util:go_default_library",
        "//pkg/util/configz:go_default_library",
        "//pkg/version:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/prometheus/client_golang/prometheus",
        "//vendor:github.com/spf13/cobra",
        "//vendor:github.com/spf13/pflag",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/util/wait",
        "//vendor:k8s.io/apiserver/pkg/server/healthz",
        "//vendor:k8s.io/apiserver/pkg/util/flag",
        "//vendor:k8s.io/client-go/discovery",
        "//vendor:k8s.io/client-go/dynamic",
        "//vendor:k8s.io/client-go/rest",
        "//vendor:k8s.io/client-go/tools/clientcmd",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["controllermanager_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//federation/pkg/federation-controller/ingress:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apiserver/pkg/util/flag",
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
        "//federation/cmd/federation-controller-manager/app/options:all-srcs",
    ],
    tags = ["automanaged"],
)
