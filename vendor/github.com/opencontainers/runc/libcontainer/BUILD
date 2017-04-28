package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "capabilities_linux.go",
        "console.go",
        "console_linux.go",
        "container.go",
        "container_linux.go",
        "criu_opts_unix.go",
        "error.go",
        "factory.go",
        "factory_linux.go",
        "generic_error.go",
        "init_linux.go",
        "message_linux.go",
        "network_linux.go",
        "notify_linux.go",
        "process.go",
        "process_linux.go",
        "restored_process.go",
        "rootfs_linux.go",
        "setgroups_linux.go",
        "setns_init_linux.go",
        "standard_init_linux.go",
        "state_linux.go",
        "stats.go",
        "stats_linux.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor/github.com/Sirupsen/logrus:go_default_library",
        "//vendor/github.com/docker/docker/pkg/mount:go_default_library",
        "//vendor/github.com/docker/docker/pkg/symlink:go_default_library",
        "//vendor/github.com/golang/protobuf/proto:go_default_library",
        "//vendor/github.com/opencontainers/runc/libcontainer/apparmor:go_default_library",
        "//vendor/github.com/opencontainers/runc/libcontainer/cgroups:go_default_library",
        "//vendor/github.com/opencontainers/runc/libcontainer/cgroups/fs:go_default_library",
        "//vendor/github.com/opencontainers/runc/libcontainer/cgroups/systemd:go_default_library",
        "//vendor/github.com/opencontainers/runc/libcontainer/configs:go_default_library",
        "//vendor/github.com/opencontainers/runc/libcontainer/configs/validate:go_default_library",
        "//vendor/github.com/opencontainers/runc/libcontainer/criurpc:go_default_library",
        "//vendor/github.com/opencontainers/runc/libcontainer/keys:go_default_library",
        "//vendor/github.com/opencontainers/runc/libcontainer/label:go_default_library",
        "//vendor/github.com/opencontainers/runc/libcontainer/seccomp:go_default_library",
        "//vendor/github.com/opencontainers/runc/libcontainer/stacktrace:go_default_library",
        "//vendor/github.com/opencontainers/runc/libcontainer/system:go_default_library",
        "//vendor/github.com/opencontainers/runc/libcontainer/user:go_default_library",
        "//vendor/github.com/opencontainers/runc/libcontainer/utils:go_default_library",
        "//vendor/github.com/syndtr/gocapability/capability:go_default_library",
        "//vendor/github.com/vishvananda/netlink:go_default_library",
        "//vendor/github.com/vishvananda/netlink/nl:go_default_library",
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
        "//vendor/github.com/opencontainers/runc/libcontainer/apparmor:all-srcs",
        "//vendor/github.com/opencontainers/runc/libcontainer/cgroups:all-srcs",
        "//vendor/github.com/opencontainers/runc/libcontainer/configs:all-srcs",
        "//vendor/github.com/opencontainers/runc/libcontainer/criurpc:all-srcs",
        "//vendor/github.com/opencontainers/runc/libcontainer/keys:all-srcs",
        "//vendor/github.com/opencontainers/runc/libcontainer/label:all-srcs",
        "//vendor/github.com/opencontainers/runc/libcontainer/seccomp:all-srcs",
        "//vendor/github.com/opencontainers/runc/libcontainer/selinux:all-srcs",
        "//vendor/github.com/opencontainers/runc/libcontainer/stacktrace:all-srcs",
        "//vendor/github.com/opencontainers/runc/libcontainer/system:all-srcs",
        "//vendor/github.com/opencontainers/runc/libcontainer/user:all-srcs",
        "//vendor/github.com/opencontainers/runc/libcontainer/utils:all-srcs",
    ],
    tags = ["automanaged"],
)
