package(default_visibility = ["//visibility:public"])

exports_files(glob(["*.txt"]))

py_test(
    name = "boilerplate_test",
    srcs = [
        "boilerplate_test.py",
        "boilerplate.py",
    ],
    data = glob(["*.txt", "test/*"]),
)
