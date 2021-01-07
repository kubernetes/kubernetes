load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "flowdistinguishermethod.go",
        "flowschema.go",
        "flowschemacondition.go",
        "flowschemaspec.go",
        "flowschemastatus.go",
        "groupsubject.go",
        "limitedprioritylevelconfiguration.go",
        "limitresponse.go",
        "nonresourcepolicyrule.go",
        "policyruleswithsubjects.go",
        "prioritylevelconfiguration.go",
        "prioritylevelconfigurationcondition.go",
        "prioritylevelconfigurationreference.go",
        "prioritylevelconfigurationspec.go",
        "prioritylevelconfigurationstatus.go",
        "queuingconfiguration.go",
        "resourcepolicyrule.go",
        "serviceaccountsubject.go",
        "subject.go",
        "usersubject.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/client-go/applyconfigurations/flowcontrol/v1beta1",
    importpath = "k8s.io/client-go/applyconfigurations/flowcontrol/v1beta1",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/api/flowcontrol/v1beta1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
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
