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

# This is a rough transliteration of build/copy-host-source.sh, using the
# Bazel workspace dependencies instead of the _output directory.

_separator = "================================================================================"

def _format_license(host_licenses_dir, pkg_name, read_cmd):
    dep_license_dir = "%s/%s" % (host_licenses_dir, pkg_name)
    dep_license_file = "%s/LICENSE" % (dep_license_dir)
    return ";".join([
      "mkdir -p %s" % (dep_license_dir),
      "echo -e '= %s licensed under: =\n' >> %s" % (pkg_name, dep_license_file),
      "%s > %s" % (read_cmd, dep_license_file),
    ])

# Creates a file named HOST_LICENSES.tar containing a LICENSES/host/... directory structure 
# containing licenses of glibc, and the Go std library and BoringCrypto / BoringSSL.
def gen_licenses(**kwargs):
    srcs = [
        "@glibc_src//:debian-copyright",
        "@go_src//file",
    ]

    # If you change this list, also be sure to change build/copy-host-source.sh.
    cmds = [
        _format_license("LICENSES/host", "glibc",    "cat $(location @glibc_src//:debian-copyright)"),
        _format_license("LICENSES/host", "go",       "tar -Oxf $(location @go_src//file) go/LICENSE"),
        _format_license("LICENSES/host", "goboring", "tar -Oxf $(location @go_src//file) go/src/crypto/internal/boring/LICENSE"),
        "tar -cf $@ --owner=0 --group=0 --numeric-owner LICENSES",
    ]

    native.genrule(
        name = "gen_licenses",
        srcs = srcs,
        outs = ["HOST_LICENSES.tar"],
        cmd = ";".join(cmds),
        **kwargs
    )
