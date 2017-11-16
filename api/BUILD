package(default_visibility = ["//visibility:public"])

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
        "//api/openapi-spec:all-srcs",
        "//api/swagger-spec:all-srcs",
    ],
    tags = ["automanaged"],
)
