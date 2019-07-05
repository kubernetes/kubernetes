load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "event.go",
        "go12.go",
        "go13.go",
        "security.go",
        "service.go",
        "sys_386.s",
        "sys_amd64.s",
    ],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/sys/windows/svc",
    importpath = "golang.org/x/sys/windows/svc",
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:windows": [
            "//vendor/golang.org/x/sys/windows:go_default_library",
        ],
        "//conditions:default": [],
    }),
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
        "//vendor/golang.org/x/sys/windows/svc/mgr:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
