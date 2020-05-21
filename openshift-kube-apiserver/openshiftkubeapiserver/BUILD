load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "flags.go",
        "patch.go",
        "patch_handlerchain.go",
        "wellknown_oauth.go",
    ],
    importpath = "k8s.io/kubernetes/openshift-kube-apiserver/openshiftkubeapiserver",
    visibility = ["//visibility:public"],
    deps = [
        "//openshift-kube-apiserver/admission/authorization/restrictusers:go_default_library",
        "//openshift-kube-apiserver/admission/authorization/restrictusers/usercache:go_default_library",
        "//openshift-kube-apiserver/admission/scheduler/nodeenv:go_default_library",
        "//openshift-kube-apiserver/enablement:go_default_library",
        "//pkg/quota/v1/generic:go_default_library",
        "//pkg/quota/v1/install:go_default_library",
        "//staging/src/k8s.io/api/authentication/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/apis/apiserver/v1alpha1:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/server:go_default_library",
        "//staging/src/k8s.io/client-go/informers:go_default_library",
        "//staging/src/k8s.io/client-go/informers/core/v1:go_default_library",
        "//staging/src/k8s.io/client-go/rest:go_default_library",
        "//staging/src/k8s.io/client-go/tools/cache:go_default_library",
        "//vendor/github.com/openshift/api/authorization/v1:go_default_library",
        "//vendor/github.com/openshift/api/config/v1:go_default_library",
        "//vendor/github.com/openshift/api/kubecontrolplane/v1:go_default_library",
        "//vendor/github.com/openshift/apiserver-library-go/pkg/admission/imagepolicy:go_default_library",
        "//vendor/github.com/openshift/apiserver-library-go/pkg/admission/imagepolicy/imagereferencemutators:go_default_library",
        "//vendor/github.com/openshift/apiserver-library-go/pkg/admission/quota/clusterresourcequota:go_default_library",
        "//vendor/github.com/openshift/apiserver-library-go/pkg/configflags:go_default_library",
        "//vendor/github.com/openshift/apiserver-library-go/pkg/securitycontextconstraints/sccadmission:go_default_library",
        "//vendor/github.com/openshift/client-go/quota/clientset/versioned:go_default_library",
        "//vendor/github.com/openshift/client-go/quota/informers/externalversions:go_default_library",
        "//vendor/github.com/openshift/client-go/quota/informers/externalversions/quota/v1:go_default_library",
        "//vendor/github.com/openshift/client-go/security/clientset/versioned:go_default_library",
        "//vendor/github.com/openshift/client-go/security/informers/externalversions:go_default_library",
        "//vendor/github.com/openshift/client-go/user/clientset/versioned:go_default_library",
        "//vendor/github.com/openshift/client-go/user/informers/externalversions:go_default_library",
        "//vendor/github.com/openshift/library-go/pkg/apiserver/admission/admissionrestconfig:go_default_library",
        "//vendor/github.com/openshift/library-go/pkg/apiserver/apiserverconfig:go_default_library",
        "//vendor/github.com/openshift/library-go/pkg/apiserver/httprequest:go_default_library",
        "//vendor/github.com/openshift/library-go/pkg/config/helpers:go_default_library",
        "//vendor/github.com/openshift/library-go/pkg/oauth/oauthdiscovery:go_default_library",
        "//vendor/github.com/openshift/library-go/pkg/quota/clusterquotamapping:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["flags_test.go"],
    embed = [":go_default_library"],
    deps = ["//vendor/github.com/openshift/api/config/v1:go_default_library"],
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
        "//openshift-kube-apiserver/openshiftkubeapiserver/paramtoken:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
