package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "admission.go",
        "authentication.go",
        "authorization.go",
        "cloudprovider.go",
        "options.go",
        "plugins.go",
        "serving.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/kubeapiserver/options",
    deps = [
        "//pkg/features:go_default_library",
        "//pkg/kubeapiserver/authenticator:go_default_library",
        "//pkg/kubeapiserver/authorizer:go_default_library",
        "//pkg/kubeapiserver/authorizer/modes:go_default_library",
        "//plugin/pkg/admission/admit:go_default_library",
        "//plugin/pkg/admission/alwayspullimages:go_default_library",
        "//plugin/pkg/admission/antiaffinity:go_default_library",
        "//plugin/pkg/admission/defaulttolerationseconds:go_default_library",
        "//plugin/pkg/admission/deny:go_default_library",
        "//plugin/pkg/admission/eventratelimit:go_default_library",
        "//plugin/pkg/admission/exec:go_default_library",
        "//plugin/pkg/admission/extendedresourcetoleration:go_default_library",
        "//plugin/pkg/admission/gc:go_default_library",
        "//plugin/pkg/admission/imagepolicy:go_default_library",
        "//plugin/pkg/admission/limitranger:go_default_library",
        "//plugin/pkg/admission/namespace/autoprovision:go_default_library",
        "//plugin/pkg/admission/namespace/exists:go_default_library",
        "//plugin/pkg/admission/noderestriction:go_default_library",
        "//plugin/pkg/admission/nodetaint:go_default_library",
        "//plugin/pkg/admission/podnodeselector:go_default_library",
        "//plugin/pkg/admission/podpreset:go_default_library",
        "//plugin/pkg/admission/podtolerationrestriction:go_default_library",
        "//plugin/pkg/admission/priority:go_default_library",
        "//plugin/pkg/admission/resourcequota:go_default_library",
        "//plugin/pkg/admission/security/podsecuritypolicy:go_default_library",
        "//plugin/pkg/admission/securitycontext/scdeny:go_default_library",
        "//plugin/pkg/admission/serviceaccount:go_default_library",
        "//plugin/pkg/admission/storage/persistentvolume/label:go_default_library",
        "//plugin/pkg/admission/storage/persistentvolume/resize:go_default_library",
        "//plugin/pkg/admission/storage/storageclass/setdefault:go_default_library",
        "//plugin/pkg/admission/storage/storageobjectinuseprotection:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/net:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission/plugin/namespace/lifecycle:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission/plugin/webhook/mutating:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission/plugin/webhook/validating:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authentication/authenticator:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/server:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/server/options:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//staging/src/k8s.io/client-go/informers:go_default_library",
        "//staging/src/k8s.io/client-go/rest:go_default_library",
        "//staging/src/k8s.io/component-base/cli/flag:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
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

go_test(
    name = "go_default_test",
    srcs = [
        "admission_test.go",
        "authentication_test.go",
        "authorization_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//pkg/kubeapiserver/authenticator:go_default_library",
        "//pkg/kubeapiserver/authorizer/modes:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/errors:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authentication/authenticator:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authentication/authenticatorfactory:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/server/options:go_default_library",
    ],
)
