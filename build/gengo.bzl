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

def _generate_prefix(tool, tool_flags, out, match):
  """Set the variables that change for different generators.

  Args:
    tool: a gengo tool, something like k8s.io/code-generator/cmd/deepcopy-gen
    tool_flags: unique flags for the tool, something like --bounding-dirs k8s.io/kubernetes
    out: the basename of the file, something like zz_generated.deepcopy.go
    match: search packages for files with this tag, something like +k8s:deepcopy-gen=
  """
  return """
  tool='$(location {tool})'
  tool_flags='{flags}'
  out='{out}'
  match='{match}'
""".format(tool=tool, tool_flags=tool_flags, out=out, match=match)

def _generate_body():
  """Main logic which runs the gengo tool on packages requesting generation.

  Usage:
    cmd = _generate_prefix(...) + _generate_body + _generate_outs(...)
  """
  return """
  #########################################
  # insert _generate_prefix() above here #
  #########################################
  # Set DEBUG=1 for extra info and checks
  DEBUG=
  # location of the prebuilt generator
  dcg="$$PWD/$$tool"

  # original pwd
  export O=$$PWD

  # split each GOPATH entry by :
  gopaths=($$(IFS=: ; echo $$GOPATH))
  # The first GOPATH entry is where go_genrule puts input source files.
  # So use this one
  srcpath="$${gopaths[0]}"

  # Use the go version bazel wants us to use, not system path
  GO="$$GOROOT/bin/go"

  # Find all packages that need generation
  files=()
  #   first, all packages with the tag, except the tool package itself
  white=($$(find . -name *.go | \
      grep -v "$${tool##*/}" | \
      (xargs grep -l "$$match" || true)))
  #   now, munge these names a bit:
  #     rename /staging/src to /vendor
  #     change ./ to k8s.io/kubernetes
  #     ignore packages that only have +k8s:foo-gen=false lines
  dirs=()
  for w in "$${white[@]}"; do
    if grep "$$match" "$$w" | grep -q -v "$${match}false"; then
      dirs+=("$$(dirname "$$w" | sed -e 's|./staging/src/|./vendor/|;s|^./|k8s.io/kubernetes/|')")
    elif [[ -n "$${DEBUG:-}" ]]; then
      echo SKIP: $$w, has only $${match}false tags
    fi
  done

  if [[ -n "$${DEBUG:-}" ]]; then
    echo "Generating: $$(IFS=$$'\n ' ; echo "$${dirs[*]}" | sort -u)"
  else
    echo "Generating $$out for $${#dirs} packages..."
  fi
  # Send the tool the comma-separate list of packages to generate
  packages="$$(IFS="," ; echo "$${dirs[*]}")"
  # Run the tool
  $$dcg \
    -v 1 \
    -i "$$packages" \
    -O "$${out%%???}" \
    -h $(location //vendor/k8s.io/code-generator/hack:boilerplate.go.txt) \
    $$tool_flags

  # DEBUG: Ensure we generated each file
  # TODO(fejta): consider deleting this
  if [[ -n "$${DEBUG:-}" ]]; then
    for p in "$${dirs[@]}"; do
      found=false
      for s in "" "k8s.io/kubernetes/vendor/" "staging/src/"; do
        if [[ -f "$$srcpath/src/$$s$$p/$$out" ]]; then
          found=true
          if [[ $$s == "k8s.io/kubernetes/vendor/" ]]; then
            grep -A 1 import $$srcpath/src/$$s$$p/$$out
          fi
          break
        fi
      done
      if [[ $$found == false ]]; then
        echo FAILED: $$p
        failed=1
      else
        echo FOUND: $$p
      fi
    done
    if [[ -n "$${failed:-}" ]]; then
      exit 1
    fi
  fi

  oifs="$$IFS"
  IFS=$$'\n'
  # use go list to create lines of pkg import import ...
  for line in $$($$GO list -f '{{.ImportPath}}{{range .Imports}} {{.}}{{end}}' "$${dirs[@]}"); do
    pkg="$${line%% *}" # first word
    imports="$${line#* }" # everything after the first word
    IFS=' '
    deps="$$srcpath/src/$$pkg/$$out.deps"
    echo > "$$deps"
    for dep in $$imports; do
      if [[ ! "$$dep" == k8s.io/kubernetes/* ]]; then
        continue
      fi
      echo //$${dep#k8s.io/kubernetes/}:go_default_library >> $$deps
    done
    IFS=$$'\n'
  done
  IFS="$$oifs"

  # copy $1 to where bazel expects to find it
  move-out() {
    D="$(OUTS)"
    dst="$$O/$${D%%/genfiles/*}/genfiles/build/$$1/$$out"
    options=(
      "k8s.io/kubernetes/$$1"
      "$${1/vendor\//}"
      "k8s.io/kubernetes/staging/src/$${1/vendor\//}"
    )
    found=false
    for o in "$${options[@]}"; do
      src="$$srcpath/src/$$o/$$out"
      if [[ ! -f "$$src" ]]; then
        continue
      fi
      found=true
      break
    done
    if [[ $$found == false ]]; then
      echo "NOT FOUND: $$1 in any of the $${options[@]}"
      exit 1
    fi

    if [[ ! -f "$$dst" ]]; then
      mkdir -p "$$(dirname "$$dst")"
      cp -f "$$src" "$$dst"
      cp -f "$$src.deps" "$$dst.deps"
      echo ... generated $$1/$$out
      return 0
    fi
  }

  ######################################
  # append _generate_outs() below here #
  ######################################
"""

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

  # script which generates all the files
  cmd = ''.join(
      # Set the variables which change for each gengo generator
      _generate_prefix(flags=flags, match=match, out=base, tool=tool),
      # Core script
      _generate_body(),
      # Copy all the generated files to their expected locations
      _generate_outs(packages),
  )

  # Bazel needs to know all the files that this rule might possibly read
  # to generate outputs.
  srcs = source_files(deps) + [
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
