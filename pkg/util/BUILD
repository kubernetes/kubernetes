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
        "doc.go",
        "template.go",
        "umask.go",
        "util.go",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "template_test.go",
        "util_test.go",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/stretchr/testify/assert",
        "//vendor:k8s.io/apimachinery/pkg/util/diff",
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
        "//pkg/util/async:all-srcs",
        "//pkg/util/bandwidth:all-srcs",
        "//pkg/util/chmod:all-srcs",
        "//pkg/util/chown:all-srcs",
        "//pkg/util/config:all-srcs",
        "//pkg/util/configz:all-srcs",
        "//pkg/util/crlf:all-srcs",
        "//pkg/util/dbus:all-srcs",
        "//pkg/util/ebtables:all-srcs",
        "//pkg/util/env:all-srcs",
        "//pkg/util/errors:all-srcs",
        "//pkg/util/exec:all-srcs",
        "//pkg/util/flock:all-srcs",
        "//pkg/util/framer:all-srcs",
        "//pkg/util/goroutinemap:all-srcs",
        "//pkg/util/hash:all-srcs",
        "//pkg/util/i18n:all-srcs",
        "//pkg/util/initsystem:all-srcs",
        "//pkg/util/interrupt:all-srcs",
        "//pkg/util/intstr:all-srcs",
        "//pkg/util/io:all-srcs",
        "//pkg/util/ipconfig:all-srcs",
        "//pkg/util/iptables:all-srcs",
        "//pkg/util/json:all-srcs",
        "//pkg/util/keymutex:all-srcs",
        "//pkg/util/labels:all-srcs",
        "//pkg/util/limitwriter:all-srcs",
        "//pkg/util/logs:all-srcs",
        "//pkg/util/maps:all-srcs",
        "//pkg/util/metrics:all-srcs",
        "//pkg/util/mount:all-srcs",
        "//pkg/util/net:all-srcs",
        "//pkg/util/netsh:all-srcs",
        "//pkg/util/node:all-srcs",
        "//pkg/util/oom:all-srcs",
        "//pkg/util/parsers:all-srcs",
        "//pkg/util/procfs:all-srcs",
        "//pkg/util/rand:all-srcs",
        "//pkg/util/resourcecontainer:all-srcs",
        "//pkg/util/rlimit:all-srcs",
        "//pkg/util/runtime:all-srcs",
        "//pkg/util/selinux:all-srcs",
        "//pkg/util/sets:all-srcs",
        "//pkg/util/slice:all-srcs",
        "//pkg/util/strings:all-srcs",
        "//pkg/util/sysctl:all-srcs",
        "//pkg/util/system:all-srcs",
        "//pkg/util/tail:all-srcs",
        "//pkg/util/taints:all-srcs",
        "//pkg/util/term:all-srcs",
        "//pkg/util/threading:all-srcs",
        "//pkg/util/uuid:all-srcs",
        "//pkg/util/validation:all-srcs",
        "//pkg/util/version:all-srcs",
        "//pkg/util/wait:all-srcs",
        "//pkg/util/workqueue/prometheus:all-srcs",
        "//pkg/util/yaml:all-srcs",
    ],
    tags = ["automanaged"],
)
