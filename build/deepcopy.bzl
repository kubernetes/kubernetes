_cmd = """
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

  # generate zz_generated.deepcopy.go
  cd $$GOPATH/src/k8s.io/kubernetes
  $$dcg \
  -v 1 \
  -i k8s.io/kubernetes/{package} \
  --bounding-dirs k8s.io/kubernetes,k8s.io/api \
  -h $(location //vendor/k8s.io/code-generator/hack:boilerplate.go.txt) \
  -O zz_generated.deepcopy

  # link it back to the expected location
  ln {package}/zz_generated.deepcopy.go $$O/'$(location zz_generated.deepcopy.go)'
"""

def cmd():
  return _cmd.format(package=native.package_name())

def k8s_deepcopy(outs):
  """genereate zz_generate.deepcopy.go for the specified package."""
  # TODO(fejta): consider running deepcopy-gen once for all packages to improve performance
  # TODO(fejta): consider auto-detecting which packages need a k8s_deepcopy rule
  native.genrule(
    name = "generate-deepcopy",
    srcs = native.glob(["**/*.go"], exclude=["zz_generated.deepcopy.go"]) + [
        "//vendor/k8s.io/code-generator/hack:boilerplate.go.txt",
        "//:all-srcs",
    ],
    outs = outs,
    tools = [
        "//vendor/k8s.io/code-generator/cmd/deepcopy-gen",
    ],
    cmd = cmd(),
    message = "generating",
  )
