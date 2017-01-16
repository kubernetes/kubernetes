package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "batch.go",
        "core.go",
        "extensions.go",
        "federation.go",
        "plugins.go",
        "server.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//federation/apis/core:go_default_library",
        "//federation/apis/core/install:go_default_library",
        "//federation/apis/core/v1:go_default_library",
        "//federation/apis/federation:go_default_library",
        "//federation/apis/federation/install:go_default_library",
        "//federation/cmd/federation-apiserver/app/options:go_default_library",
        "//federation/registry/cluster/etcd:go_default_library",
        "//pkg/admission:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/api/install:go_default_library",
        "//pkg/api/rest:go_default_library",
        "//pkg/apis/batch:go_default_library",
        "//pkg/apis/batch/install:go_default_library",
        "//pkg/apis/extensions:go_default_library",
        "//pkg/apis/extensions/install:go_default_library",
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//pkg/cloudprovider/providers:go_default_library",
        "//pkg/controller/informers:go_default_library",
        "//pkg/generated/openapi:go_default_library",
        "//pkg/genericapiserver:go_default_library",
        "//pkg/genericapiserver/filters:go_default_library",
        "//pkg/kubeapiserver:go_default_library",
        "//pkg/kubeapiserver/admission:go_default_library",
        "//pkg/registry/batch/job/storage:go_default_library",
        "//pkg/registry/cachesize:go_default_library",
        "//pkg/registry/core/configmap/storage:go_default_library",
        "//pkg/registry/core/event/storage:go_default_library",
        "//pkg/registry/core/namespace/storage:go_default_library",
        "//pkg/registry/core/secret/storage:go_default_library",
        "//pkg/registry/core/service/storage:go_default_library",
        "//pkg/registry/extensions/daemonset/storage:go_default_library",
        "//pkg/registry/extensions/deployment/storage:go_default_library",
        "//pkg/registry/extensions/ingress/storage:go_default_library",
        "//pkg/registry/extensions/replicaset/storage:go_default_library",
        "//pkg/registry/generic:go_default_library",
        "//pkg/registry/generic/registry:go_default_library",
        "//pkg/routes:go_default_library",
        "//pkg/version:go_default_library",
        "//plugin/pkg/admission/admit:go_default_library",
        "//plugin/pkg/admission/deny:go_default_library",
        "//plugin/pkg/admission/gc:go_default_library",
        "//plugin/pkg/admission/namespace/lifecycle:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/pborman/uuid",
        "//vendor:github.com/spf13/cobra",
        "//vendor:github.com/spf13/pflag",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/util/errors",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apimachinery/pkg/util/wait",
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
        "//federation/cmd/federation-apiserver/app/options:all-srcs",
    ],
    tags = ["automanaged"],
)
