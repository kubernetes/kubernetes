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
        "client.go",
        "config.go",
        "mesos.go",
        "plugins.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/resource:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/cloudprovider:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/util/net:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/mesos/mesos-go/detector",
        "//vendor:github.com/mesos/mesos-go/detector/zoo",
        "//vendor:github.com/mesos/mesos-go/mesosproto",
        "//vendor:golang.org/x/net/context",
        "//vendor:gopkg.in/gcfg.v1",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "client_test.go",
        "config_test.go",
        "mesos_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/cloudprovider:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/util/net:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/mesos/mesos-go/detector",
        "//vendor:github.com/mesos/mesos-go/mesosutil",
        "//vendor:golang.org/x/net/context",
    ],
)
