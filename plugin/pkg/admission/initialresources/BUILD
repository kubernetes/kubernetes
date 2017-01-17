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
        "//pkg/api/resource:go_default_library",
        "//pkg/client/restclient:go_default_library",
        "//pkg/client/unversioned/clientcmd:go_default_library",
        "//vendor:cloud.google.com/go/compute/metadata",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/hawkular/hawkular-client-go/metrics",
        "//vendor:github.com/influxdata/influxdb/client",
        "//vendor:golang.org/x/oauth2",
        "//vendor:golang.org/x/oauth2/google",
        "//vendor:google.golang.org/api/cloudmonitoring/v2beta2",
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "admission_test.go",
        "hawkular_test.go",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/admission:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/api/resource:go_default_library",
        "//vendor:github.com/stretchr/testify/require",
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
)
