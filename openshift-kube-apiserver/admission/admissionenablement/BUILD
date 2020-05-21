load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "admission.go",
        "admission_config.go",
        "register.go",
    ],
    importpath = "k8s.io/kubernetes/openshift-kube-apiserver/admission/admissionenablement",
    visibility = ["//visibility:public"],
    deps = [
        "//cmd/kube-apiserver/app/options:go_default_library",
        "//openshift-kube-apiserver/admission/authorization/restrictusers:go_default_library",
        "//openshift-kube-apiserver/admission/autoscaling/clusterresourceoverride:go_default_library",
        "//openshift-kube-apiserver/admission/autoscaling/runonceduration:go_default_library",
        "//openshift-kube-apiserver/admission/customresourcevalidation/customresourcevalidationregistration:go_default_library",
        "//openshift-kube-apiserver/admission/namespaceconditions:go_default_library",
        "//openshift-kube-apiserver/admission/network/externalipranger:go_default_library",
        "//openshift-kube-apiserver/admission/network/restrictedendpoints:go_default_library",
        "//openshift-kube-apiserver/admission/route:go_default_library",
        "//openshift-kube-apiserver/admission/scheduler/nodeenv:go_default_library",
        "//openshift-kube-apiserver/admission/scheduler/podnodeconstraints:go_default_library",
        "//plugin/pkg/admission/resourcequota:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission/plugin/webhook/mutating:go_default_library",
        "//staging/src/k8s.io/client-go/informers:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//vendor/github.com/openshift/apiserver-library-go/pkg/admission/imagepolicy:go_default_library",
        "//vendor/github.com/openshift/apiserver-library-go/pkg/admission/imagepolicy/apis/imagepolicy/v1:go_default_library",
        "//vendor/github.com/openshift/apiserver-library-go/pkg/admission/quota/clusterresourcequota:go_default_library",
        "//vendor/github.com/openshift/apiserver-library-go/pkg/securitycontextconstraints/sccadmission:go_default_library",
        "//vendor/github.com/openshift/library-go/pkg/apiserver/admission/admissiontimeout:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["register_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//openshift-kube-apiserver/admission/customresourcevalidation/customresourcevalidationregistration:go_default_library",
        "//pkg/kubeapiserver/options:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/server:go_default_library",
        "//vendor/github.com/openshift/library-go/pkg/apiserver/admission/admissionregistrationtesting:go_default_library",
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
    visibility = ["//visibility:public"],
)
