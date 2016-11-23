package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
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
        "//pkg/api/errors:go_default_library",
        "//pkg/api/unversioned:go_default_library",
        "//pkg/client/unversioned/clientcmd:go_default_library",
        "//pkg/client/unversioned/clientcmd/api:go_default_library",
        "//pkg/kubectl:go_default_library",
        "//pkg/kubectl/cmd:go_default_library",
        "//pkg/kubectl/cmd/templates:go_default_library",
        "//pkg/kubectl/cmd/util:go_default_library",
        "//pkg/kubectl/resource:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/util/flag:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/spf13/cobra",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "join_test.go",
        "unjoin_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//federation/apis/federation:go_default_library",
        "//federation/apis/federation/v1beta1:go_default_library",
        "//federation/pkg/kubefed/testing:go_default_library",
        "//federation/pkg/kubefed/util:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/api/errors:go_default_library",
        "//pkg/api/testapi:go_default_library",
        "//pkg/api/unversioned:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/client/restclient/fake:go_default_library",
        "//pkg/client/typed/dynamic:go_default_library",
        "//pkg/client/unversioned/clientcmd:go_default_library",
        "//pkg/client/unversioned/clientcmd/api:go_default_library",
        "//pkg/kubectl/cmd/testing:go_default_library",
        "//pkg/kubectl/cmd/util:go_default_library",
        "//vendor:k8s.io/client-go/pkg/util/diff",
    ],
)
