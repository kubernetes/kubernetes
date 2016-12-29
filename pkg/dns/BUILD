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
        "dns.go",
        "doc.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/dns/config:go_default_library",
        "//pkg/dns/treecache:go_default_library",
        "//pkg/dns/util:go_default_library",
        "//pkg/util/validation:go_default_library",
        "//pkg/util/wait:go_default_library",
        "//vendor:github.com/coreos/etcd/client",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/miekg/dns",
        "//vendor:github.com/skynetservices/skydns/msg",
        "//vendor:k8s.io/client-go/kubernetes",
        "//vendor:k8s.io/client-go/pkg/api/v1",
        "//vendor:k8s.io/client-go/pkg/apis/meta/v1",
        "//vendor:k8s.io/client-go/pkg/runtime",
        "//vendor:k8s.io/client-go/pkg/watch",
        "//vendor:k8s.io/client-go/tools/cache",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["dns_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
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
        "//vendor:k8s.io/client-go/kubernetes/fake",
        "//vendor:k8s.io/client-go/pkg/api/v1",
        "//vendor:k8s.io/client-go/pkg/apis/meta/v1",
        "//vendor:k8s.io/client-go/tools/cache",
    ],
)
