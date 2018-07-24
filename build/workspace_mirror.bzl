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

prefix = "https://storage.googleapis.com/k8s-bazel-cache/"

def mirror(url):
    """Try downloading a URL from a GCS mirror first, then from the original.

    Update the GCS bucket using bazel run //hack:update-mirror"""
    return [prefix + url, url]

def mirror_urls():
    # This function only gives proper results when executed from WORKSPACE,
    # but the data is needed in sh_binary, which can only be in a BUILD file.
    # Thus, it is be exported by a repository_rule (which executes in WORKSPACE)
    # to be used by the sh_binary.
    urls = []
    for k, v in native.existing_rules().items():
        us = list(v.get("urls", []))
        if "url" in v:
            us.append(v["url"])
        for u in us:
            if u and not u.startswith(prefix):
                urls.append(u)
    return sorted(urls)

def export_urls_impl(repo_ctx):
    repo_ctx.file(repo_ctx.path("BUILD.bazel"), """
exports_files(glob(["**"]), visibility=["//visibility:public"])
""")
    repo_ctx.file(
        repo_ctx.path("urls.txt"),
        # Add a trailing newline, since the "while read" loop needs it
        content = ("\n".join(repo_ctx.attr.urls) + "\n"),
    )

_export_urls = repository_rule(
    attrs = {
        "urls": attr.string_list(mandatory = True),
    },
    local = True,
    implementation = export_urls_impl,
)

def export_urls(name):
    return _export_urls(name = name, urls = mirror_urls())
