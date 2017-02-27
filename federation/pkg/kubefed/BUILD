package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "join.go",
        "kubefed.go",
        "unjoin.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//federation/apis/federation:go_default_library",
        "//federation/pkg/kubefed/init:go_default_library",
        "//federation/pkg/kubefed/util:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/apis/extensions:go_default_library",
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//pkg/kubectl:go_default_library",
        "//pkg/kubectl/cmd:go_default_library",
        "//pkg/kubectl/cmd/templates:go_default_library",
        "//pkg/kubectl/cmd/util:go_default_library",
        "//pkg/kubectl/resource:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/spf13/cobra",
        "//vendor:github.com/spf13/pflag",
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apiserver/pkg/util/flag",
        "//vendor:k8s.io/client-go/tools/clientcmd",
        "//vendor:k8s.io/client-go/tools/clientcmd/api",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "join_test.go",
        "unjoin_test.go",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//federation/apis/federation:go_default_library",
        "//federation/apis/federation/v1beta1:go_default_library",
        "//federation/pkg/kubefed/testing:go_default_library",
        "//federation/pkg/kubefed/util:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/api/testapi:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/extensions/v1beta1:go_default_library",
        "//pkg/kubectl/cmd/testing:go_default_library",
        "//pkg/kubectl/cmd/util:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/api/equality",
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/util/diff",
        "//vendor:k8s.io/client-go/dynamic",
        "//vendor:k8s.io/client-go/rest/fake",
        "//vendor:k8s.io/client-go/tools/clientcmd",
        "//vendor:k8s.io/client-go/tools/clientcmd/api",
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
        "//federation/pkg/kubefed/init:all-srcs",
        "//federation/pkg/kubefed/testing:all-srcs",
        "//federation/pkg/kubefed/util:all-srcs",
    ],
    tags = ["automanaged"],
)
