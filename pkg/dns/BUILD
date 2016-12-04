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
        "dns.go",
        "doc.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1:go_default_library",
        "//pkg/api/v1/endpoints:go_default_library",
        "//pkg/apis/meta/v1:go_default_library",
        "//pkg/client/cache:go_default_library",
        "//pkg/client/clientset_generated/release_1_5:go_default_library",
        "//pkg/dns/config:go_default_library",
        "//pkg/dns/treecache:go_default_library",
        "//pkg/dns/util:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/util/validation:go_default_library",
        "//pkg/util/wait:go_default_library",
        "//pkg/watch:go_default_library",
        "//vendor:github.com/coreos/etcd/client",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/miekg/dns",
        "//vendor:github.com/skynetservices/skydns/msg",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["dns_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1:go_default_library",
        "//pkg/api/v1/endpoints:go_default_library",
        "//pkg/apis/meta/v1:go_default_library",
        "//pkg/client/cache:go_default_library",
        "//pkg/client/clientset_generated/release_1_5/fake:go_default_library",
        "//pkg/dns/config:go_default_library",
        "//pkg/dns/treecache:go_default_library",
        "//pkg/dns/util:go_default_library",
        "//pkg/util/sets:go_default_library",
        "//vendor:github.com/coreos/etcd/client",
        "//vendor:github.com/miekg/dns",
        "//vendor:github.com/skynetservices/skydns/msg",
        "//vendor:github.com/skynetservices/skydns/server",
        "//vendor:github.com/stretchr/testify/assert",
        "//vendor:github.com/stretchr/testify/require",
    ],
)
