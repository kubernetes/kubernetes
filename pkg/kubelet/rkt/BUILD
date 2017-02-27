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
        "cap.go",
        "config.go",
        "container_id.go",
        "doc.go",
        "image.go",
        "log.go",
        "rkt.go",
        "systemd.go",
        "version.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1:go_default_library",
        "//pkg/credentialprovider:go_default_library",
        "//pkg/kubelet/container:go_default_library",
        "//pkg/kubelet/events:go_default_library",
        "//pkg/kubelet/images:go_default_library",
        "//pkg/kubelet/leaky:go_default_library",
        "//pkg/kubelet/lifecycle:go_default_library",
        "//pkg/kubelet/network:go_default_library",
        "//pkg/kubelet/network/hairpin:go_default_library",
        "//pkg/kubelet/prober/results:go_default_library",
        "//pkg/kubelet/types:go_default_library",
        "//pkg/kubelet/util/format:go_default_library",
        "//pkg/securitycontext:go_default_library",
        "//pkg/util/exec:go_default_library",
        "//pkg/util/parsers:go_default_library",
        "//pkg/util/selinux:go_default_library",
        "//pkg/util/strings:go_default_library",
        "//pkg/util/term:go_default_library",
        "//pkg/util/version:go_default_library",
        "//vendor:github.com/appc/spec/schema",
        "//vendor:github.com/appc/spec/schema/types",
        "//vendor:github.com/coreos/go-systemd/dbus",
        "//vendor:github.com/coreos/go-systemd/unit",
        "//vendor:github.com/coreos/rkt/api/v1alpha",
        "//vendor:github.com/docker/engine-api/types",
        "//vendor:github.com/golang/glog",
        "//vendor:golang.org/x/net/context",
        "//vendor:google.golang.org/grpc",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/errors",
        "//vendor:k8s.io/apimachinery/pkg/util/uuid",
        "//vendor:k8s.io/apimachinery/pkg/util/wait",
        "//vendor:k8s.io/client-go/tools/record",
        "//vendor:k8s.io/client-go/util/flowcontrol",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "fake_rkt_interface_test.go",
        "rkt_test.go",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1:go_default_library",
        "//pkg/kubelet/container:go_default_library",
        "//pkg/kubelet/container/testing:go_default_library",
        "//pkg/kubelet/lifecycle:go_default_library",
        "//pkg/kubelet/network:go_default_library",
        "//pkg/kubelet/network/kubenet:go_default_library",
        "//pkg/kubelet/network/testing:go_default_library",
        "//pkg/kubelet/types:go_default_library",
        "//pkg/util/exec:go_default_library",
        "//vendor:github.com/appc/spec/schema",
        "//vendor:github.com/appc/spec/schema/types",
        "//vendor:github.com/coreos/go-systemd/dbus",
        "//vendor:github.com/coreos/rkt/api/v1alpha",
        "//vendor:github.com/golang/mock/gomock",
        "//vendor:github.com/stretchr/testify/assert",
        "//vendor:golang.org/x/net/context",
        "//vendor:google.golang.org/grpc",
        "//vendor:k8s.io/apimachinery/pkg/api/resource",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/errors",
        "//vendor:k8s.io/client-go/util/testing",
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
