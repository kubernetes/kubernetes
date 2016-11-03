package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "semver.go",
        "version.go",
        ":base_generated",
    ],
    deps = [
        "//vendor:github.com/blang/semver",
        "//vendor:github.com/golang/glog",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["semver_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [],
)

# this file replace base.go in this package and is integrated into the
# bazel build.
genrule(
    name = "base_generated",
    outs = [
        "base_generated.go",
    ],
    cmd = """
set -o errexit
set -o nounset
set -o pipefail

eval $$(cat bazel-out/volatile-status.txt | tr ' ' '=')

cat <<EOF > $@
package version

import "time"

var (
  gitVersion   string = "$${BUILD_SCM_REVISION}"
  gitCommit    string = "$${BUILD_GIT_COMMIT}"
  gitTreeState string = "$${BUILD_SCM_STATUS}"
  buildDate    string = time.Unix($${BUILD_TIMESTAMP}/1000, 0).Format(time.RFC3339)
  // deprecated
  gitMajor string = "$${BUILD_MAJOR_VERSION}"
  gitMinor string = "$${BUILD_MINOR_VERSION}"
)
EOF
    """,
    stamp = True,
)
