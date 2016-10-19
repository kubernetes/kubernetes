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
        "admission.go",
        "data_source.go",
        "gcm.go",
        "hawkular.go",
        "influxdb.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/admission:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/api/errors:go_default_library",
        "//pkg/api/resource:go_default_library",
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//pkg/client/restclient:go_default_library",
        "//pkg/client/unversioned/clientcmd:go_default_library",
        "//vendor:cloud.google.com/go/compute/metadata",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/hawkular/hawkular-client-go/metrics",
        "//vendor:github.com/influxdata/influxdb/client",
        "//vendor:golang.org/x/oauth2",
        "//vendor:golang.org/x/oauth2/google",
        "//vendor:google.golang.org/api/cloudmonitoring/v2beta2",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "admission_test.go",
        "hawkular_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/admission:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/api/resource:go_default_library",
        "//vendor:github.com/stretchr/testify/require",
    ],
)
