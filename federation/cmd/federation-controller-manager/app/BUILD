package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "controllermanager.go",
        "plugins.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//federation/client/clientset_generated/federation_release_1_5:go_default_library",
        "//federation/cmd/federation-controller-manager/app/options:go_default_library",
        "//federation/pkg/dnsprovider:go_default_library",
        "//federation/pkg/dnsprovider/providers/aws/route53:go_default_library",
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
        "//pkg/apis/meta/v1:go_default_library",
        "//pkg/client/restclient:go_default_library",
        "//pkg/client/typed/discovery:go_default_library",
        "//pkg/client/unversioned/clientcmd:go_default_library",
        "//pkg/healthz:go_default_library",
        "//pkg/runtime/schema:go_default_library",
        "//pkg/util/config:go_default_library",
        "//pkg/util/configz:go_default_library",
        "//pkg/util/wait:go_default_library",
        "//pkg/version:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/prometheus/client_golang/prometheus",
        "//vendor:github.com/spf13/cobra",
        "//vendor:github.com/spf13/pflag",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["controllermanager_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//federation/pkg/federation-controller/ingress:go_default_library",
        "//pkg/apis/meta/v1:go_default_library",
        "//pkg/runtime/schema:go_default_library",
        "//pkg/util/config:go_default_library",
    ],
)
