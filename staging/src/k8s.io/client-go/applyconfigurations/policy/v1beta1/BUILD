load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "allowedcsidriver.go",
        "allowedflexvolume.go",
        "allowedhostpath.go",
        "eviction.go",
        "fsgroupstrategyoptions.go",
        "hostportrange.go",
        "idrange.go",
        "poddisruptionbudget.go",
        "poddisruptionbudgetspec.go",
        "poddisruptionbudgetstatus.go",
        "podsecuritypolicy.go",
        "podsecuritypolicyspec.go",
        "runasgroupstrategyoptions.go",
        "runasuserstrategyoptions.go",
        "runtimeclassstrategyoptions.go",
        "selinuxstrategyoptions.go",
        "supplementalgroupsstrategyoptions.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/client-go/applyconfigurations/policy/v1beta1",
    importpath = "k8s.io/client-go/applyconfigurations/policy/v1beta1",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/api/policy/v1beta1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/intstr:go_default_library",
        "//staging/src/k8s.io/client-go/applyconfigurations/core/v1:go_default_library",
        "//staging/src/k8s.io/client-go/applyconfigurations/meta/v1:go_default_library",
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
