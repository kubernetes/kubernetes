load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "blame.go",
        "common.go",
        "doc.go",
        "options.go",
        "references.go",
        "remote.go",
        "repository.go",
        "status.go",
        "submodule.go",
        "worktree.go",
        "worktree_commit.go",
        "worktree_status.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:darwin_amd64": [
            "worktree_darwin.go",
        ],
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "worktree_linux.go",
        ],
        "@io_bazel_rules_go//go/platform:windows_amd64": [
            "worktree_windows.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "gopkg.in/src-d/go-git.v4",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/sergi/go-diff/diffmatchpatch:go_default_library",
        "//vendor/gopkg.in/src-d/go-billy.v3:go_default_library",
        "//vendor/gopkg.in/src-d/go-billy.v3/osfs:go_default_library",
        "//vendor/gopkg.in/src-d/go-billy.v3/util:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/config:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/internal/revision:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/filemode:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/format/gitignore:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/format/index:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/format/packfile:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/object:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/protocol/packp:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/protocol/packp/capability:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/protocol/packp/sideband:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/revlist:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/storer:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/transport:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing/transport/client:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/storage:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/storage/filesystem:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/storage/memory:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/utils/diff:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/utils/ioutil:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/utils/merkletrie:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/utils/merkletrie/filesystem:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/utils/merkletrie/index:go_default_library",
        "//vendor/gopkg.in/src-d/go-git.v4/utils/merkletrie/noder:go_default_library",
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
        "//vendor/gopkg.in/src-d/go-git.v4/config:all-srcs",
        "//vendor/gopkg.in/src-d/go-git.v4/internal/revision:all-srcs",
        "//vendor/gopkg.in/src-d/go-git.v4/plumbing:all-srcs",
        "//vendor/gopkg.in/src-d/go-git.v4/storage:all-srcs",
        "//vendor/gopkg.in/src-d/go-git.v4/utils/binary:all-srcs",
        "//vendor/gopkg.in/src-d/go-git.v4/utils/diff:all-srcs",
        "//vendor/gopkg.in/src-d/go-git.v4/utils/ioutil:all-srcs",
        "//vendor/gopkg.in/src-d/go-git.v4/utils/merkletrie:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
