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

# Genrule wrapper around the go-bindata utility.
# IMPORTANT: Any changes to this rule may also require changes to hack/generate-bindata.sh.
def go_bindata(
    name, srcs, outs,
    compress=True,
    include_metadata=True,
    pkg="generated",
    ignores=["\.jpg", "\.png", "\.md", "BUILD(\.bazel)?"],
    **kw):

  args = []
  for ignore in ignores:
    args.extend(["-ignore", "'%s'" % ignore])
  if not include_metadata:
    args.append("-nometadata")
  if not compress:
    args.append("-nocompress")

  native.genrule(
    name = name,
    srcs = srcs,
    outs = outs,
    cmd = """
    $(location //vendor/github.com/jteeuwen/go-bindata/go-bindata:go-bindata) \
      -o "$@" -pkg %s -prefix $$(pwd) %s $(SRCS)
    """ % (pkg, " ".join(args)),
    tools = [
      "//vendor/github.com/jteeuwen/go-bindata/go-bindata",
    ],
    **kw
  )
