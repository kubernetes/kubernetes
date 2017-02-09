package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "plugins.go",
        "server.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//cmd/kube-apiserver/app/options:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/apis/batch:go_default_library",
        "//pkg/capabilities:go_default_library",
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//pkg/cloudprovider:go_default_library",
        "//pkg/cloudprovider/providers:go_default_library",
        "//pkg/controller/informers:go_default_library",
        "//pkg/controller/serviceaccount:go_default_library",
        "//pkg/generated/openapi:go_default_library",
        "//pkg/kubeapiserver:go_default_library",
        "//pkg/kubeapiserver/admission:go_default_library",
        "//pkg/kubeapiserver/authenticator:go_default_library",
        "//pkg/master:go_default_library",
        "//pkg/master/tunneler:go_default_library",
        "//pkg/registry/cachesize:go_default_library",
        "//pkg/version:go_default_library",
        "//plugin/pkg/admission/admit:go_default_library",
        "//plugin/pkg/admission/alwayspullimages:go_default_library",
        "//plugin/pkg/admission/antiaffinity:go_default_library",
        "//plugin/pkg/admission/deny:go_default_library",
        "//plugin/pkg/admission/exec:go_default_library",
        "//plugin/pkg/admission/gc:go_default_library",
        "//plugin/pkg/admission/imagepolicy:go_default_library",
        "//plugin/pkg/admission/initialresources:go_default_library",
        "//plugin/pkg/admission/limitranger:go_default_library",
        "//plugin/pkg/admission/namespace/autoprovision:go_default_library",
        "//plugin/pkg/admission/namespace/exists:go_default_library",
        "//plugin/pkg/admission/namespace/lifecycle:go_default_library",
        "//plugin/pkg/admission/persistentvolume/label:go_default_library",
        "//plugin/pkg/admission/podnodeselector:go_default_library",
        "//plugin/pkg/admission/resourcequota:go_default_library",
        "//plugin/pkg/admission/security/podsecuritypolicy:go_default_library",
        "//plugin/pkg/admission/securitycontext/scdeny:go_default_library",
        "//plugin/pkg/admission/serviceaccount:go_default_library",
        "//plugin/pkg/admission/storageclass/default:go_default_library",
        "//vendor:github.com/go-openapi/spec",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/pborman/uuid",
        "//vendor:github.com/spf13/cobra",
        "//vendor:github.com/spf13/pflag",
        "//vendor:k8s.io/apimachinery/pkg/openapi",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/util/errors",
        "//vendor:k8s.io/apimachinery/pkg/util/net",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apimachinery/pkg/util/wait",
        "//vendor:k8s.io/apiserver/pkg/admission",
        "//vendor:k8s.io/apiserver/pkg/server",
        "//vendor:k8s.io/apiserver/pkg/server/filters",
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
        "//cmd/kube-apiserver/app/options:all-srcs",
    ],
    tags = ["automanaged"],
)
