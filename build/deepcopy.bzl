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

load("@io_kubernetes_build//defs:go.bzl", "go_genrule")

_generate_prefix = """
  tool='$(location {tool})' # something like k8s.io/code-generator/cmd/deepcopy-gen
  tool_flags='{flags}' # something like --bounding-dirs k8s.io/kubernetes,k8s.io/api
  out='{out}'  # something like zz_generated.deepcopy.go
  match='{match}'  # something like +k8s:deepcopy-gen=
"""

_generate_body = """
  # location of the prebuilt generator
  dcg="$$PWD/$$tool"

  # original pwd
  export O=$$PWD

  # split each GOPATH entry by :
  gopaths=($$(IFS=: ; echo $$GOPATH))

  srcpath="$${gopaths[0]}"
  dstpath="$${gopaths[1]}"
  if [[ "$$dstpath" != "$$GENGOPATH" ]]; then
    env | sort
    echo "Envionrmental assumptions failed: GENGOPATH is no the second GOPATH"
    exit 1
  fi

  # when vendor/k8s.io/foo symlinks to staging/src/k8s.io/foo
  # bazel will wind up creating concrete versions of both folders,
  # putting half the files in staging and the other half in vendor
  rsync --links --recursive staging/src/k8s.io/ vendor/k8s.io/

  # Find all packages that request generation, except for the tool itself
  files=()
  GO="$$GOROOT/bin/go"
  white=($$(find . -name *.go | \
      grep -v "$${tool##*/}" | \
      (xargs grep -l "$$match" || true)))
  dirs=()
  for w in "$${white[@]}"; do
    if grep "$$match" "$$w" | grep -q -v "$${match}false"; then
      dirs+=("$$(dirname "$$w" | sed -e 's|./staging/src/|./vendor/|;s|^./|k8s.io/kubernetes/|')")
    else
      echo SKIP: $$w, has only $${match}false tags
    fi
  done

  echo "Generating: $${dirs[*]}"
  packages="$$(IFS="," ; echo "$${dirs[*]}")"  # Create comma-separated list of packages expected by tool
  $$dcg \
    -v 1 \
    -i "$$packages" \
    -O "$${out%%???}" \
    -h $(location //vendor/k8s.io/code-generator/hack:boilerplate.go.txt) \
    $$tool_flags

  # Ensure we generated each file
  DEBUG=1
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
  for line in $$(go list -f '{{.ImportPath}}{{range .Imports}} {{.}}{{end}}' "$${dirs[@]}"); do
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

  # detect if the out file does not exist or changed
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

  ####################################
  # append move-out calls below here #
  ####################################
"""

def k8s_deepcopy_all(name, packages):
  """Generate zz_generated.deepcopy.go for all specified packages in one invocation."""
  k8s_gengo_all(
    name=name,
    base="zz_generated.deepcopy.go",
    tool="//vendor/k8s.io/code-generator/cmd/deepcopy-gen",
    match="+k8s:deepcopy-gen=",
    flags="--bounding-dirs k8s.io/kubernetes,k8s.io/api",
    packages=packages,
  )

def k8s_deepcopy(outs):
  """find the zz_generated.deepcopy.go for the calling package."""
  k8s_gengo(
    name="deepcopy",
    outs=outs,
  )

def k8s_defaulter_all(name, packages):
  """Generate zz_generated.defaults.go for all specified packages in one invocation."""
  k8s_gengo_all(
    name=name,
    base="zz_generated.defaults.go",
    tool="//vendor/k8s.io/code-generator/cmd/defaulter-gen",
    match="+k8s:defaulter-gen=",
    flags="--extra-peer-dirs %s" % ",".join(["k8s.io/kubernetes/%s" % p for p in packages]),
    packages=packages,
  )

def k8s_defaulter(outs):
  """find the zz_generated.defaulter.go for the calling package."""
  k8s_gengo(
    name="defaulter",
    outs=outs,
  )

def k8s_conversion_all(name, packages):
  k8s_gengo_all(
    name=name,
    base="zz_generated.conversion.go",
    tool="//vendor/k8s.io/code-generator/cmd/conversion-gen",
    match="+k8s:conversion-gen=",
    flags="--extra-peer-dirs %s" % ",".join([
        "k8s.io/kubernetes/pkg/apis/core",
	"k8s.io/kubernetes/pkg/apis/core/v1",
	"k8s.io/api/core/v1",
    ]),
    packages=packages)

def k8s_conversion(outs):
  k8s_gengo(
    name="conversion",
    outs=outs,
  )

def k8s_gengo_all(name, base, tool, flags, match, packages):
  """Use TOOL to generate BASE files in all the PACKAGES with a MATCH comment."""
  # Tell bazel all the files we will generate
  go_files = ["%s/%s" % (p, base) for p in packages] # generated files
  dep_files = ["%s.deps" % g for g in go_files] # list of new packages dependencies
  outs = go_files + dep_files

  # script which generates all the files
  cmd = _generate_prefix.format(
    flags=flags,
    match=match,
    out=base,
    tool=tool,
  ) + _generate_body + "\n".join(["move-out %s" % p for p in packages])


  # Rule which generates a set of out files given a set of input src and tool files
  go_genrule(
    name = name,
    # Bazel needs to know all the files that this rule might possibly read
    srcs = [
        "//vendor/k8s.io/code-generator/hack:boilerplate.go.txt",
        "//:all-srcs", # TODO(fejta): consider updating kazel to provide just the list of go files
	"@go_sdk//:files",  # k8s.io/gengo expects to be able to read $GOROOT/src
    ],
    # Build the tool we run to generate the files
    tools = [tool],
    # Tell bazel all the files we will generate
    outs = outs,
    # script bazel runs to generate the files
    cmd = cmd,
    # command-line message to display
    message = "Generating %s files in %d packages for" % (base, len(packages)),
  )

def go_package_name():
  """Return path/in/k8s or vendor/k8s.io/repo/path"""
  name = native.package_name()
  if name.startswith('staging/src/'):
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
