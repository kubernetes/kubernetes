#!/usr/bin/env python

# Copyright 2015 The Kubernetes Authors All rights reserved.
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

import subprocess
from path import path


class KubernetesInstaller():
    """
    This class contains the logic needed to install kuberentes binary files.
    """

    def __init__(self, arch, version, master, output_dir):
        """ Gather the required variables for the install. """
        # The kubernetes charm needs certain commands to be aliased.
        self.aliases = {'kube-proxy': 'proxy',
                        'kubelet': 'kubelet'}
        self.arch = arch
        self.version = version
        self.master = master
        self.output_dir = output_dir

    def download(self):
        """ Download the kuberentes binaries from the kubernetes master. """
        url = 'http://{0}/kubernetes/{1}/local/bin/linux/{2}'.format(
            self.master, self.version, self.arch)
        if not self.output_dir.isdir():
            self.output_dir.makedirs_p()

        for key in self.aliases:
            uri = '{0}/{1}'.format(url, key)
            destination = self.output_dir / key
            wget = 'wget -nv {0} -O {1}'.format(uri, destination)
            print(wget)
            output = subprocess.check_output(wget.split())
            print(output)
            destination.chmod(0o755)

    def install(self, install_dir=path('/usr/local/bin')):
        """ Create links to the binary files to the install directory. """

        if not install_dir.isdir():
            install_dir.makedirs_p()

        # Create the symbolic links to the real kubernetes binaries.
        for key, value in self.aliases.iteritems():
            target = self.output_dir / key
            if target.exists():
                link = install_dir / value
                if link.exists():
                    link.remove()
                target.symlink(link)
            else:
                print('Error target file {0} does not exist.'.format(target))
                exit(1)
