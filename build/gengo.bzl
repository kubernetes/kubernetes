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

# Bazel macros for using the k8s.io/gengo framework to generate zz_generated.FOO.go files.
#
# Usage:
#   1) Create a rule that builds all the generated files, in //build for example
#       load("//build:gengo.bzl", "k8s_gengo_all")
#       k8s_gengo_all(
#           name="deepcopy-sources",
#           base="zz_generated.deepcopy.go",
#           tool="//vendor/k8s.io/code-generator/cmd/deepcopy-gen",
#           match="+k8s:deepcopy-gen=",
#           flags="--bounding-dirs k8s.io/kubernetes,k8s.io/api",
#           packages=[
#             "k8s.io/kubernetes/every/package",
#             "k8s.io/kubernetes/that/wants",
#             "k8s.io/kubernetes/a/zz_genrated.deepcopy.go/file",
#           ],
#           deps=[
#             "vendor/github.com/every/package",
#             "vendor/whatever.io/used/by/packages",
#           ],
#      )
#   2) For every package in the packages list, add the following:
#       load("//build:gengo.bzl", "k8s_gengo")
#       k8s_gengo(name="generate-deepcopy", outs=["zz_generated.deepcopy.go"])
#   3) Be sure to use the //build:go.bzl wrapped version of go_library
#       load("@io_bazel_rules_go//go:def.bzl", "go_library")  # NO: generate-deepcopy fails
#       load("//build:go.bzl", "go_library")  # YES: generate-deepcopy succeeds
#   4) gazelle will now notice the existence of this file and add it to srcs
#       go_library(
#           name="go_default_library",
#           srcs=[..., "zz_generated.deepcopy.go"],
#           ...
#       )

load("@io_kubernetes_build//defs:go.bzl", _go_genrule="go_genrule")

def _generate_prefix(tool, tool_flags, out, match, header):
  """Set the variables that change for different generators.

  Args:
    tool: a gengo tool, something like k8s.io/code-generator/cmd/deepcopy-gen
    tool_flags: unique flags for the tool, something like --bounding-dirs k8s.io/kubernetes
    out: the basename of the file, something like zz_generated.deepcopy.go
    match: search packages for files with this tag, something like +k8s:deepcopy-gen=
    header: boilerplate file, something like vendor/k8s.io/code-generator/hack/boilerplate.go.txt
  """
  return """
  tool='$(location {tool})'
  tool_flags='{flags}'
  out='{out}'
  match='{match}'
  head='{header}'
""".format(tool=tool, tool_flags=tool_flags, out=out, match=match)

def _generate_outs(packages):
  """Return the move-out commands which copy files to where bazel expects."""
  return "\n".join(["move-out '%s'" % p for p in packages])

def source_files(pkgs):
  """Return a list of sourcefiles targets for the specified packages.

  The :go_default_library.sourcefiles target includes all the
  non-generated *.go files for the package. Both the gengo and go list
  tools use these files to reflect over the package.
  """
  return ["//%s:go_default_library.sourcefiles" % p for p in pkgs]

def k8s_gengo_all(name, base, tool, flags, match, packages, deps):
  """Use TOOL to generate BASE files in all the PACKAGES with a MATCH comment."""
  # TODO(fejta): gazelle doesn't appear to auto-generate rules for this package
  packages = [p for p in packages if not p.startswith("vendor/k8s.io/code-generator/_examples/")]
  # Add any missing packages to deps
  deps = {d: True for d in (deps + packages) if not d.startswith("vendor/k8s.io/code-generator/_examples/")}.keys()

  # Tell bazel all the files we will generate
  go_files = ["%s/%s" % (p, base) for p in packages] # generated files
  dep_files = ["%s.deps" % g for g in go_files] # list of new packages dependencies
  outs = go_files + dep_files
  header="$(location //vendor/k8s.io/code-generator/hack:boilerplate.go.txt)"

  # script which generates all the files
  cmd = '\n'.join(
      # Set the variables which change for each gengo generator
      _generate_prefix(flags=flags, match=match, out=base, tool=tool, header=header),
      # Core script
      "STARTED_FROM_GENGO_BZL=true; . '$(location :gengo.sh)'",
      # Copy all the generated files to their expected locations
      _generate_outs(packages),
  )

  # Bazel needs to know all the files that this rule might possibly read
  # to generate outputs.
  srcs = source_files(deps) + [
      ":gengo.sh",
      "//vendor/k8s.io/code-generator/hack:boilerplate.go.txt", # header boilerplate
      "@go_sdk//:files",  # k8s.io/gengo expects to be able to read $GOROOT/src
  ]
  msg = "Generating %s files in %d packages for" % (base, len(packages))

  # Rule which generates a set of out files given a set of input src and tool files
  _go_genrule(
      name=name,
      srcs=srcs, # input source files used to generate outputs
      tools = [tool], # Build the gengo tool the script runs to generate outputs
      outs=outs, # Tell bazel all the files we will generate
      cmd=cmd, # script bazel runs to generate the files
      message=msg, # command-line message to display
  )

def go_package_name():
  """Return path/in/k8s or vendor/k8s.io/repo/path"""
  name = native.package_name()
  if name.startswith('staging/src/'): # We actually want to use vendor/
    return name.replace('staging/src/', 'vendor/')
  return name

def k8s_gengo(name, outs):
  """Find the zz_generated.NAME.go file for the package which calls this macro."""
  # TODO(fejta): consider auto-detecting which packages need a k8s_deepcopy rule

  # Ensure outs is correct, we only accept this as an arg so gazelle knows about the file
  if len(outs) != 1:
    fail("outs must contain exactly 1 item:" % outs)
  out = outs[0]
  if not out.endswith(".go"):
    fail("outs must end with .go: %s" % outs)
  sources = "//build:%s-sources" % name
  native.genrule(
    name = "generate-%s" % name,
    srcs = [sources],
    outs = outs,
    cmd = """
    # The file we want to find
    goal="{package}/{out}"

    # Places we might find it aka //build:something-sources
    options=($(locations {sources}))

    # Iterate through these places
    for o in "$${{options[@]}}"; do
      if [[ "$$o" != */"$$goal" ]]; then
        continue  # not here
      fi

      mkdir -p "$$(dirname "$$goal")"
      cp -f "$$o" "$(location {out})"
      exit 0
    done
    echo "MISSING: could not find $$goal in any of the $${{options[@]}}"
    exit 1
    """.format(
      package=go_package_name(),
      out=out,
      sources=sources,
    ),
    message = "Extracting generated zz_generated.deepcopy.go for",
  )
