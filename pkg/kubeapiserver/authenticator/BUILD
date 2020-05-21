package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "config.go",
        "patch_authenticator.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/kubeapiserver/authenticator",
    deps = [
        "//openshift-kube-apiserver/admission/authorization/restrictusers/usercache:go_default_library",
        "//openshift-kube-apiserver/admission/customresourcevalidation/oauth:go_default_library",
        "//openshift-kube-apiserver/authentication/oauth:go_default_library",
        "//openshift-kube-apiserver/enablement:go_default_library",
        "//pkg/features:go_default_library",
        "//pkg/serviceaccount:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/net:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authentication/authenticator:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authentication/authenticatorfactory:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authentication/group:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authentication/request/anonymous:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authentication/request/bearertoken:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authentication/request/headerrequest:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authentication/request/union:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authentication/request/websocket:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authentication/request/x509:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authentication/token/cache:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authentication/token/tokenfile:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authentication/token/union:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/server:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/server/dynamiccertificates:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//staging/src/k8s.io/apiserver/plugin/pkg/authenticator/password/passwordfile:go_default_library",
        "//staging/src/k8s.io/apiserver/plugin/pkg/authenticator/request/basicauth:go_default_library",
        "//staging/src/k8s.io/apiserver/plugin/pkg/authenticator/token/oidc:go_default_library",
        "//staging/src/k8s.io/apiserver/plugin/pkg/authenticator/token/webhook:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//staging/src/k8s.io/client-go/plugin/pkg/client/auth:go_default_library",
        "//staging/src/k8s.io/client-go/tools/cache:go_default_library",
        "//staging/src/k8s.io/client-go/util/keyutil:go_default_library",
        "//vendor/github.com/go-openapi/spec:go_default_library",
        "//vendor/github.com/openshift/client-go/oauth/clientset/versioned:go_default_library",
        "//vendor/github.com/openshift/client-go/oauth/informers/externalversions:go_default_library",
        "//vendor/github.com/openshift/client-go/user/clientset/versioned:go_default_library",
        "//vendor/github.com/openshift/client-go/user/informers/externalversions:go_default_library",
        "//vendor/github.com/openshift/library-go/pkg/authentication/bootstrapauthenticator:go_default_library",
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
