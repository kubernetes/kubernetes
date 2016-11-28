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
        "garbagecollector.go",
        "metrics.go",
        "rate_limiter_helper.go",
        "uid_cache.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/errors:go_default_library",
        "//pkg/api/meta:go_default_library",
        "//pkg/api/meta/metatypes:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/meta/v1:go_default_library",
        "//pkg/client/cache:go_default_library",
        "//pkg/client/typed/dynamic:go_default_library",
        "//pkg/controller/garbagecollector/metaonly:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/runtime/schema:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/util/clock:go_default_library",
        "//pkg/util/errors:go_default_library",
        "//pkg/util/metrics:go_default_library",
        "//pkg/util/runtime:go_default_library",
        "//pkg/util/sets:go_default_library",
        "//pkg/util/wait:go_default_library",
        "//pkg/util/workqueue:go_default_library",
        "//pkg/watch:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/golang/groupcache/lru",
        "//vendor:github.com/prometheus/client_golang/prometheus",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["garbagecollector_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api/install:go_default_library",
        "//pkg/api/meta/metatypes:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/apimachinery/registered:go_default_library",
        "//pkg/apis/meta/v1:go_default_library",
        "//pkg/client/restclient:go_default_library",
        "//pkg/client/typed/dynamic:go_default_library",
        "//pkg/controller/garbagecollector/metaonly:go_default_library",
        "//pkg/runtime/schema:go_default_library",
        "//pkg/runtime/serializer:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/util/clock:go_default_library",
        "//pkg/util/json:go_default_library",
        "//pkg/util/sets:go_default_library",
        "//pkg/util/workqueue:go_default_library",
        "//vendor:github.com/stretchr/testify/assert",
    ],
)
