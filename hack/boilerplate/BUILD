package(default_visibility = ["//visibility:public"])

exports_files(glob(["*.txt"]))

py_test(
    name = "boilerplate_test",
    srcs = [
        "boilerplate.py",
        "boilerplate_test.py",
    ],
    data = glob([
        "*.txt",
        "test/*",
    ]),
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
        "//hack/boilerplate/test:all-srcs",
    ],
    tags = ["automanaged"],
)
