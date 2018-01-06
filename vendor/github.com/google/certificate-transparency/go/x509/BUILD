load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "cert_pool.go",
        "pem_decrypt.go",
        "pkcs1.go",
        "pkcs8.go",
        "root.go",
        "sec1.go",
        "verify.go",
        "x509.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:darwin": [
            "root_darwin.go",
            "root_stub.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "root_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "root_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "root_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "root_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "root_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "root_plan9.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "root_windows.go",
        ],
        "//conditions:default": [],
    }),
    cgo = True,
    clinkopts = select({
        "@io_bazel_rules_go//go/platform:darwin": [
            "-framework CoreFoundation -framework Security",
        ],
        "//conditions:default": [],
    }),
    copts = select({
        "@io_bazel_rules_go//go/platform:darwin": [
            "-mmacosx-version-min=10.6 -D__MAC_OS_X_VERSION_MAX_ALLOWED=1060",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/google/certificate-transparency/go/x509",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/google/certificate-transparency/go/asn1:go_default_library",
        "//vendor/github.com/google/certificate-transparency/go/x509/pkix:go_default_library",
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
        "//vendor/github.com/google/certificate-transparency/go/x509/pkix:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
