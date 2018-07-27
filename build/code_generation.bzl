# Copyright 2018 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@bazel_skylib//:lib.bzl", "paths")

# The go_prefix for this repository, needed to map relative package paths into
# a full-resolved go package.
go_prefix = "k8s.io/kubernetes"

def _format_output_dict(id, tags, output):
    """Pretty-prints the provided tags dict as a Skylark variable into output.

    All keys and values will be sorted so that the resulting output is stable.

    Args:
      id: The identifier to use in the output.
      tags: A dict->dict->dict, which will be converted to a dict->dict->list
          and then pretty-printed in output.
      output: A list of output lines into which this method appends.
    """
    output.append("%s = {" % id)

    for tag in sorted(tags.keys()):
        d = tags[tag]
        output.append("%s%s: {" % (4 * " ", repr(tag)))
        for k in sorted(d.keys()):
            output.append("%s%s: [" % (8 * " ", repr(k)))
            for v in sorted(d[k].keys()):
                output.append("%s%s," % (12 * " ", repr(v)))
            output.append("%s]," % (8 * " "))

        output.append("%s}," % (4 * " "))

    output.append("}\n")

def _line_to_pkg_tag_values(workspace_root, line):
    """Parses a grep output line.

    The resulting go package name, k8s codegen tag, and tag values will be
    returned as a struct (with fields "pkg", "tag", and "values" respectively).

    Returns None if this line should be skipped, e.g. examples directories which
    do not have BUILD files generated.

    Args:
      workspace_root: The path to the workspace root.
          It is expected that all lines will begin with this as a prefix.
      line: The raw line from grep. All trailing whitespace should be removed.
    """
    # Each line looks something like
    # /a/full/path/to/file.go:+k8s:foo=bar

    # First split out the filename from the matched tag blog.
    # TODO: figure out how to handle Windows drive letters
    fname, _, match = line.partition(":")

    # Remove the workspace_root from the path, yielding just a relative path,
    # and then treat the resulting dirname as the go package name.
    remove_prefix_len = len(str(workspace_root))
    if workspace_root != "/":
        # Remove the slash too
        remove_prefix_len += 1
    pkg = paths.dirname(fname[remove_prefix_len:])

    # Skip things like _examples packages, since gazelle doesn't generate
    # rules in these packages.
    if pkg.startswith("_") or "/_" in pkg:
        return None

    # Strip off the leading "+k8s:", then split into tag name and value.
    tag, _, parsed_value = match.partition(":")[2].partition("=")

    # The value may be multiple values separated by a comma, e.g. +k8s:foo=bar,baz,
    # so split the values.
    values = parsed_value.split(",")

    return struct(pkg = pkg, tag = tag, values = values)

def _find_generator_tag_pkgs_impl(repo_ctx):
    workspace_root = repo_ctx.path(repo_ctx.attr._workspace).dirname

    result = repo_ctx.execute(
        ["grep", "--color=never", "-Hrox", "--include=*.go", "\\s*//\s*+k8s:\\S*=\\S*\\s*", workspace_root],
        quiet = True,
    )
    if result.return_code:
        fail("failed searching for generator build tags: %s" % result.stderr)

    # Maps tag names -> go packages -> values found for that tag in that package
    tags_pkgs_values = {}

    # Maps tag names -> values found for that tag -> go packages with those tag/value mappings
    tags_values_pkgs = {}

    for line in result.stdout.splitlines():
        parsed = _line_to_pkg_tag_values(workspace_root, line)
        if not parsed:  # this pkg should be skipped
            continue

        for value in parsed.values:
            # Since Skylark doesn't have sets, we take the Go approach and fake a set using a dictionary
            tags_pkgs_values.setdefault(parsed.tag, default = {}).setdefault(parsed.pkg, default = {})[value] = None
            tags_values_pkgs.setdefault(parsed.tag, default = {}).setdefault(value, default = {})[parsed.pkg] = None

    output = []
    _format_output_dict("tags_pkgs_values", tags_pkgs_values, output)
    _format_output_dict("tags_values_pkgs", tags_values_pkgs, output)

    repo_ctx.file(
        "tags.bzl",
        content = "\n".join(output),
        executable = False,
    )

    repo_ctx.file(
        "BUILD.bazel",
        content = "exports_files(glob([\"*.bzl\"]))",
        executable = False,
    )

    # Ensure that this rule always runs by touching the WORKSPACE file
    repo_ctx.execute(["touch", repo_ctx.path(repo_ctx.attr._workspace)])

_find_generator_tag_pkgs = repository_rule(
    attrs = {
        "_workspace": attr.label(
            allow_single_file = True,
            default = "@//:WORKSPACE",
        ),
    },
    local = True,
    implementation = _find_generator_tag_pkgs_impl,
)

def find_generator_tag_pkgs(name = "io_k8s_code_generation", **kw):
    """Finds all k8s codegen tags, creating dicts in a tags.bzl file.

    Kubernetes code generation uses comments in Go source files that look
    like Go build tags. For example, a Go source file containing
    // +k8s:foo=bar
    would indicate that the package containing this file should enable the "foo"
    generator. Sometimes the value is used by generator; in other cases, this
    may be a boolean indicating whether the generator is enabled.

    This rule implementation uses grep to find all Go source files containing
    anything that looks like a k8s codegen tag.

    It processes this output to create two dictionaries, saved in a tags.bzl
    file:
      tags_pkgs_values: maps tag name -> packages using that tag ->
          list of values for that tag in that package.
      tags_values_pkgs: maps tag name -> values found for that tag ->
          list of packages with that tag-value combination.

    tags_values_pkgs is likely to be more useful in code generation, since it
    clearly indicates which packages should be included.

    For example, to find all packages requesting OpenAPI generation
    (using +k8s:openapi-gen=true), one could use

    load("@io_k8s_code_generation//:tags.bzl", "tags_values_pkgs")
    pkgs = tags_values_pkgs["openapi-gen"]["true"]
    """
    _find_generator_tag_pkgs(name = name, **kw)

def bazel_go_library(pkg):
    """Returns the Bazel label for the Go library for the provided package.

    This is intended to be used with the @io_k8s_code_generation//:tags.bzl dictionaries; for example:

    load("@io_k8s_code_generation//:tags.bzl", "tags_values_pkgs")
    some_rule(
        ...
        deps = [bazel_go_library(pkg) for pkg in tags_values_pkgs["openapi-gen"]["true"]],
        ...
    )
    """
    return "//%s:go_default_library" % pkg

def go_pkg(pkg):
    """Returns the full Go package name for the provided workspace-relative package.

    This is suitable to pass to tools depending on the Go build library.
    If any packages are in staging/src, they are remapped to their intended path in vendor/.
    This is intended to be used with the @io_k8s_code_generation//:tags.bzl dictionaries.
    For example:

    genrule(
        ...
        cmd = "do something --pkgs=%s" % ",".join([go_pkg(pkg) for pkg in tags_values_pkgs["openapi-gen"]["true"]]),
        ...
    )
    """
    return go_prefix + "/" + pkg.replace("staging/src/", "vendor/", maxsplit = 1)

# We want several helper functions to be private, but then we can't access them for unit testing.
# Export them through a public struct, but make it clear they should only be used for unit testing.
exported_for_unit_testing = struct(
    _format_output_dict = _format_output_dict,
    _line_to_pkg_tag_values = _line_to_pkg_tag_values,
)
