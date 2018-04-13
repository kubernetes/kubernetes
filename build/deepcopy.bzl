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

_generate = """
  # location of prebuilt deepcopy generator
  dcg=$$PWD/'$(location //vendor/k8s.io/code-generator/cmd/deepcopy-gen)'

  # original pwd
  export O=$$PWD

  # gopath/goroot for genrule
  export GOPATH=$$PWD/._go
  export GOROOT=/usr/lib/google-golang

  # symlink in source to new gopath
  # TODO(fejta): figure out what subset of this nonsense is necessary
  mkdir -p $$GOPATH/src/k8s.io
  ln -snf $$PWD $$GOPATH/src/k8s.io/kubernetes
  touch $$GOPATH/BUILD.bazel # avoid cycle of the above symlink
  for i in $$(ls vendor/k8s.io); do
    ln -snf $$PWD/vendor/k8s.io/$$i $$GOPATH/src/k8s.io/$$i
    ln -snf $$PWD/vendor/k8s.io/$$i $$GOPATH/src/k8s.io/kubernetes/vendor/k8s.io/$$i
  done
  # symlink in all the staging dirs
  for i in $$(ls staging/src/k8s.io); do
    ln -snf $$PWD/staging/src/k8s.io/$$i $$GOPATH/src/k8s.io/$$i
    ln -snf $$PWD/staging/src/k8s.io/$$i $$GOPATH/src/k8s.io/kubernetes/vendor/k8s.io/$$i
  done

  cd $$GOPATH/src/k8s.io/kubernetes

  # Find all packages that request deepcopy-generation, except for the deepcopy-gen tool itself
  files=()
  for p in $$(find . -name *.go | \
      (xargs grep -l '+k8s:deepcopy-gen=' || true) | \
      (xargs -n 1 dirname || true) | sort -u | \
      sed -e 's|./staging/src/||' | xargs go list | grep -v k8s.io/code-generator/cmd/deepcopy-gen); do
    files+=("$$p")
  done
  packages="$$(IFS="," ; echo "$${files[*]}")"  # Create comma-separated list of packages expected by tool
  echo "Generating: $${files[*]}..."
  $$dcg \
  -v 1 \
  -i "$$packages" \
  --bounding-dirs k8s.io/kubernetes,k8s.io/api \
  -h $(location //vendor/k8s.io/code-generator/hack:boilerplate.go.txt) \
  -O zz_generated.deepcopy

  for p in "$${files[@]}"; do
    found=false
    for s in "" "k8s.io/kubernetes/vendor/" "staging/src/"; do
      if [[ -f "$$GOPATH/src/$$s$$p/zz_generated.deepcopy.go" ]]; then
        found=true
        if [[ $$s == "k8s.io/kubernetes/vendor/" ]]; then
          grep -A 1 import $$GOPATH/src/$$s$$p/zz_generated.deepcopy.go
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

  # detect if the out file does not exist or changed
  move-out() {
    dst="$$O/$(@D)/$$1/zz_generated.deepcopy.go"
    options=(
      "k8s.io/kubernetes/$$1"
      "$${1/vendor\//}"
      "k8s.io/kubernetes/staging/src/$${1/vendor\//}"
    )
    found=false
    for o in "$${options[@]}"; do
      src="$$GOPATH/src/$$o/zz_generated.deepcopy.go"
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
    echo SRC: $$src
    echo DST: $$dst
    if [[ ! -f "$$src" ]]; then
     echo PROBLEM $$src
     ls $$(dirname $$src)
     echo odd
     exit 1
    fi

    if [[ ! -f "$$dst" ]]; then
      echo "MISSING: $$dst"
      mkdir -p "$$(dirname "$$dst")"
      ln "$$src" "$$dst"
    elif ! cmp -s "$$src" "$$dst"; then
      # link it back to the expected location
      echo "UPDATE: $$dst (src $$? old)"
      ln -f "$$src" "$$dst"
    else
      echo "GOOD NEWS: using cached version of $$dst"
    fi
  }
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

def go_package_name():
  """Return path/in/k8s or vendor/k8s.io/repo/path"""
  name = native.package_name()
  if name.startswith('staging/src/'):
    return name.replace('staging/src/', 'vendor/')
  return name

def k8s_deepcopy_all(name, packages):
  """Generate zz_generated.deepcopy.go for all specified packages in one invocation."""
  # Tell bazel all the files we will generate
  outs = ["%s/zz_generated.deepcopy.go" % p for p in packages]
  # script which generates all the files
  cmd = _generate + '\n'.join(['move-out %s' % p for p in packages])

  # Rule which generates a set of out files given a set of input src and tool files
  native.genrule(
    name = name,
    # Bazel needs to know all the files that this rule might possibly read
    srcs = [
        "//vendor/k8s.io/code-generator/hack:boilerplate.go.txt",
        "//:all-srcs", # TODO(fejta): consider updating kazel to provide just the list of go files
    ],
    # Build the tool we run to generate the files
    tools = ["//vendor/k8s.io/code-generator/cmd/deepcopy-gen"],
    # Tell bazel all the files we will generate
    outs = outs,
    # script bazel runs to generate the files
    cmd = cmd,
    # command-line message to display
    message = "Generating %d zz_generated.deepcopy.go files for" % len(packages),
  )

def k8s_deepcopy(outs):
  """Find the zz_generate.deepcopy.go file for the package which calls this macro."""
  # TODO(fejta): consider auto-detecting which packages need a k8s_deepcopy rule

  # Ensure outs is correct, we only accept this as an arg so gazelle knows about the file
  if outs != ["zz_generated.deepcopy.go"]:
    fail("outs must equal [\"zz_genereated.deepcopy.go\"], not %s" % outs)
  native.genrule(
    name = "generate-deepcopy",
    srcs = [
        "//build:deepcopy-sources",
    ],
    outs = outs,
    tools = [
        "//vendor/k8s.io/code-generator/cmd/deepcopy-gen",
    ],
    cmd = """
    # The file we want to find
    goal="{package}/zz_generated.deepcopy.go"

    # Places we might find it
    options=($(locations //build:deepcopy-sources))

    # Iterate through these places
    for o in "$${{options[@]}}"; do
      if [[ "$$o" != */"$$goal" ]]; then
        continue  # not here
      fi

      echo "FOUND: $$goal at $$o"
      mkdir -p "$$(dirname "$$goal")"
      ln -f "$$o" "$(location zz_generated.deepcopy.go)"
      exit 0
    done
    echo "MISSING: could not find $$goal in any of the $${{options[@]}}"
    exit 1
    """.format(package=go_package_name()),
    message = "Extracting generated zz_generated.deepcopy.go for",
  )
