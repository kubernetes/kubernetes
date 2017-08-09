package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["servicecontroller.go"],
    deps = [
        "//federation/apis/federation:go_default_library",
        "//federation/apis/federation/v1beta1:go_default_library",
        "//federation/client/clientset_generated/federation_clientset:go_default_library",
        "//federation/pkg/federation-controller/service/ingress:go_default_library",
        "//federation/pkg/federation-controller/util:go_default_library",
        "//federation/pkg/federation-controller/util/clusterselector:go_default_library",
        "//federation/pkg/federation-controller/util/deletionhelper:go_default_library",
        "//federation/pkg/federation-controller/util/eventsink:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/controller:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/watch:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes:go_default_library",
        "//vendor/k8s.io/client-go/listers/core/v1:go_default_library",
        "//vendor/k8s.io/client-go/tools/cache:go_default_library",
        "//vendor/k8s.io/client-go/tools/record:go_default_library",
        "//vendor/k8s.io/client-go/util/flowcontrol:go_default_library",
        "//vendor/k8s.io/client-go/util/workqueue:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["servicecontroller_test.go"],
    library = ":go_default_library",
    deps = [
        "//federation/apis/federation/v1beta1:go_default_library",
        "//federation/client/clientset_generated/federation_clientset/fake:go_default_library",
        "//federation/pkg/federation-controller/service/ingress:go_default_library",
        "//federation/pkg/federation-controller/util:go_default_library",
        "//federation/pkg/federation-controller/util/deletionhelper:go_default_library",
        "//federation/pkg/federation-controller/util/test:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/github.com/stretchr/testify/require:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/fake:go_default_library",
        "//vendor/k8s.io/client-go/listers/core/v1:go_default_library",
        "//vendor/k8s.io/client-go/tools/cache:go_default_library",
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
        "//federation/pkg/federation-controller/service/dns:all-srcs",
        "//federation/pkg/federation-controller/service/ingress:all-srcs",
    ],
    tags = ["automanaged"],
)
