load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "cert_pool.go",
        "error.go",
        "errors.go",
        "names.go",
        "pem_decrypt.go",
        "pkcs1.go",
        "pkcs8.go",
        "revoked.go",
        "root.go",
        "sec1.go",
        "verify.go",
        "x509.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:darwin": [
            "root_darwin.go",
            "root_nocgo_darwin.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "root_bsd.go",
            "root_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "root_bsd.go",
            "root_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "root_linux.go",
            "root_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "root_nacl.go",
            "root_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "root_bsd.go",
            "root_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "root_bsd.go",
            "root_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "root_plan9.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "root_solaris.go",
            "root_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "root_windows.go",
        ],
        "//conditions:default": [],
    }) + select({
        "@io_bazel_rules_go//go/platform:darwin_386": [
            "nilref_nil_darwin.go",
            "nilref_zero_darwin.go",
            "root_cgo_darwin.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin_amd64": [
            "nilref_nil_darwin.go",
            "nilref_zero_darwin.go",
            "root_cgo_darwin.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin_arm": [
            "root_darwin_armx.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin_arm64": [
            "root_darwin_armx.go",
        ],
        "//conditions:default": [],
    }),
    cgo = True,
    clinkopts = select({
        "@io_bazel_rules_go//go/platform:darwin_386": [
            "-framework CoreFoundation -framework Security",
        ],
        "@io_bazel_rules_go//go/platform:darwin_amd64": [
            "-framework CoreFoundation -framework Security",
        ],
        "//conditions:default": [],
    }),
    copts = select({
        "@io_bazel_rules_go//go/platform:darwin_386": [
            "-mmacosx-version-min=10.6 -D__MAC_OS_X_VERSION_MAX_ALLOWED=1080",
        ],
        "@io_bazel_rules_go//go/platform:darwin_amd64": [
            "-mmacosx-version-min=10.6 -D__MAC_OS_X_VERSION_MAX_ALLOWED=1080",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/google/certificate-transparency-go/x509",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/google/certificate-transparency-go/asn1:go_default_library",
        "//vendor/github.com/google/certificate-transparency-go/tls:go_default_library",
        "//vendor/github.com/google/certificate-transparency-go/x509/pkix:go_default_library",
        "//vendor/golang.org/x/crypto/cryptobyte:go_default_library",
        "//vendor/golang.org/x/crypto/cryptobyte/asn1:go_default_library",
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
        "//vendor/github.com/google/certificate-transparency-go/x509/pkix:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
