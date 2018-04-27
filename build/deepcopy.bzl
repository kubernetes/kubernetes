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

def k8s_deepcopy_all(name, packages, deps):
  """Generate zz_generated.deepcopy.go for all specified packages in one invocation."""
  k8s_gengo_all(
    name=name,
    base="zz_generated.deepcopy.go",
    tool="//vendor/k8s.io/code-generator/cmd/deepcopy-gen",
    match="+k8s:deepcopy-gen=",
    flags="--bounding-dirs k8s.io/kubernetes,k8s.io/api",
    packages=packages,
    deps=deps,
  )

def k8s_deepcopy(outs):
  """find the zz_generated.deepcopy.go for the calling package."""
  k8s_gengo(
    name="deepcopy",
    outs=outs,
  )

def k8s_defaulter_all(name, packages, deps):
  """Generate zz_generated.defaults.go for all specified packages in one invocation."""
  k8s_gengo_all(
    name=name,
    base="zz_generated.defaults.go",
    tool="//vendor/k8s.io/code-generator/cmd/defaulter-gen",
    match="+k8s:defaulter-gen=",
    flags="--extra-peer-dirs %s" % ",".join(["k8s.io/kubernetes/%s" % p for p in packages]),
    packages=packages,
    deps=deps,
  )

def k8s_defaulter(outs):
  """find the zz_generated.defaulter.go for the calling package."""
  k8s_gengo(
    name="defaulter",
    outs=outs,
  )

def k8s_conversion_all(name, packages, deps):
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
    packages=packages,
    deps=deps,
    )

def k8s_conversion(outs):
  k8s_gengo(
    name="conversion",
    outs=outs,
  )
