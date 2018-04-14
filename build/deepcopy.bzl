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

_generate = """
  tool='$(location {tool})' # something like k8s.io/code-generator/cmd/deepcopy-gen
  tool_flags='{flags}' # something like --bounding-dirs k8s.io/kubernetes,k8s.io/api
  out='{out}'  # something like zz_generated.deepcopy.go
  match='{match}'  # something like +k8s:deepcopy-gen=

  # location of the prebuilt generator
  dcg="$$PWD/$$tool"

  # original pwd
  export O=$$PWD

  # split each GOPATH entry by :
  gopaths=($$(IFS=: ; echo $$GOPATH))

  srcpath="$${{gopaths[0]}}"
  dstpath="$${{gopaths[1]}}"
  if [[ "$$dstpath" != "$$GENGOPATH" ]]; then
    env | sort
    echo "Envionrmental assumptions failed: GENGOPATH is no the second GOPATH"
    exit 1
  fi

  # Ensure all the vendor/staging code is in the expected locations
  for i in $$(ls vendor/k8s.io); do
    ln -snf $$PWD/vendor/k8s.io/$$i $$dstpath/src/k8s.io/$$i
  done
  # symlink in all the staging dirs
  for i in $$(ls staging/src/k8s.io); do
    ln -snf $$PWD/staging/src/k8s.io/$$i $$dstpath/src/k8s.io/$$i
  done


  # Find all packages that request generation, except for the tool itself
  files=()
  GO="$$GOROOT/bin/go"
  for p in $$(find . -name *.go | \
      (xargs grep -l "$$match" || true) | \
      (xargs -n 1 dirname || true) | sort -u | \
      sed -e 's|./staging/src/||' | xargs "$$GO" list | grep -v "$${{tool##*/}}"); do
    files+=("$$p")
  done
  packages="$$(IFS="," ; echo "$${{files[*]}}")"  # Create comma-separated list of packages expected by tool
  echo "Generating: $${{files[*]}}..."
  $$dcg \
    -v 1 \
    -i "$$packages" \
    -O "$${{out%%???}}" \
    -h $(location //vendor/k8s.io/code-generator/hack:boilerplate.go.txt) \
    $$tool_flags

  # Ensure we generated each file
  DEBUG=1
  if [[ -n "$${{DEBUG:-}}" ]]; then
    for p in "$${{files[@]}}"; do
      found=false
      for s in "" "k8s.io/kubernetes/vendor/" "staging/src/"; do
        echo $$s
        if [[ -f "$$srcpath/src/$$s$$p/$$out" ]]; then
          found=true
          if [[ $$s == "k8s.io/kubernetes/vendor/" ]]; then
            grep -A 1 import $$srcpath/src/$$s$$p/$$out
          fi
          echo FOUND: $$s$$p
          break
        fi
      done
      if [[ $$found == false ]]; then
        echo FAILED: $$p
        exit 1
      fi
    done
  fi

  # detect if the out file does not exist or changed
  move-out() {{
    D="$(OUTS)"
    dst="$$O/$${{D%%/genfiles/*}}/genfiles/build/$$1/$$out"
    options=(
      "k8s.io/kubernetes/$$1"
      "$${{1/vendor\//}}"
      "k8s.io/kubernetes/staging/src/$${{1/vendor\//}}"
    )
    found=false
    for o in "$${{options[@]}}"; do
      src="$$srcpath/src/$$o/$$out"
      if [[ ! -f "$$src" ]]; then
        continue
      fi
      found=true
      break
    done
    if [[ $$found == false ]]; then
      echo "NOT FOUND: $$1 in any of the $${{options[@]}}"
      exit 1
    fi

    if [[ ! -f "$$dst" ]]; then
      mkdir -p "$$(dirname "$$dst")"
      cp -f "$$src" "$$dst"
      echo ... generated $$1/$$out
      return 0
    fi
  }}

  ####################################
  # append move-out calls below here #
  ####################################
"""

_link = """
  # location of prebuilt deepcopy generator
  dcg=$$PWD/'$(location //vendor/k8s.io/code-generator/cmd/deepcopy-gen)'
  # original pwd
  export O=$$PWD
  # gopath/goroot for genrule
  export GOPATH=$$PWD/.go
  export GOROOT=/usr/lib/google-golang

  # symlink in source into new gopath
  mkdir -p $$GOPATH/src/k8s.io
  ln -snf $$PWD $$GOPATH/src/k8s.io/kubernetes
  # symlink in all the staging dirs
  for i in $$(ls staging/src/k8s.io); do
    ln -snf $$PWD/staging/src/k8s.io/$$i $$GOPATH/src/k8s.io/$$i
  done
  # prevent symlink recursion
  touch $$GOPATH/BUILD.bazel

  echo GP: {go_package}
  echo BP: {bazel_package}
  # generate zz_generated.deepcopy.go
  cd $$GOPATH/src/k8s.io/kubernetes
  $$dcg \
  -v 1 \
  -i {go_package} \
  --bounding-dirs k8s.io/kubernetes,k8s.io/api \
  -h $(location //vendor/k8s.io/code-generator/hack:boilerplate.go.txt) \
  -O zz_generated.deepcopy

  # detect if the out file does not exist or changed
  out="$$O/$(location zz_generated.deepcopy.go)"
  now="{bazel_package}/zz_generated.deepcopy.go"
  if [[ ! -f "$$out" ]]; then
    echo "NEW: $$out, linking in..."
    ln "$$now" "$$out"
  elif ! cmp -s "$$now" "$$old"; then
    # link it back to the expected location
    echo "UPDATE: $$out (now $$? old), updating..."
    ln "$$now" "$$out"
  else
    echo "CACHED: using cached version of $$out"
  fi
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

def k8s_gengo_all(name, base, tool, flags, match, packages):
  """Use TOOL to generate BASE files in all the PACKAGES with a MATCH comment."""
  # Tell bazel all the files we will generate
  outs = ["%s/%s" % (p, base) for p in packages]

  # script which generates all the files
  cmd= _generate.format(
    flags=flags,
    match=match,
    out=base,
    packages=packages,
    tool=tool,
  ) + "\n".join(["move-out %s" % p for p in packages])


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

      echo "FOUND: $$goal at $$o"
      mkdir -p "$$(dirname "$$goal")"
      ln -f "$$o" "$(location {out})"
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
