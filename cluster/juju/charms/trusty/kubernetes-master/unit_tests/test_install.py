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

from mock import patch, Mock, MagicMock
from path import Path
import pytest
import sys

# Munge the python path so we can find our hook code
d = Path('__file__').parent.abspath() / 'hooks'
sys.path.insert(0, d.abspath())

# Import the modules from the hook
import install

class TestInstallHook():

    @patch('install.path')
    def test_update_rc_files(self, pmock):
        """
        Test happy path on updating env files. Assuming everything
        exists and is in place.
        """
        pmock.return_value.lines.return_value =  ['line1', 'line2']
        install.update_rc_files(['test1', 'test2'])
        pmock.return_value.write_lines.assert_called_with(['line1', 'line2',
                                                           'test1', 'test2'])

    def test_update_rc_files_with_nonexistant_path(self):
        """
        Test an unhappy path if the bashrc/users do not exist.
        """
        with pytest.raises(OSError) as exinfo:
            install.update_rc_files(['test1','test2'])

    @patch('install.fetch')
    @patch('install.hookenv')
    def test_package_installation(self, hemock, ftmock):
        """
        Verify we are calling the known essentials to build and syndicate
        kubes.
        """
        pkgs = ['build-essential', 'git',
                'make', 'nginx', 'python-pip']
        install.install_packages()
        hemock.log.assert_called_with('Installing Debian packages')
        ftmock.filter_installed_packages.assert_called_with(pkgs)

    @patch('install.archiveurl.ArchiveUrlFetchHandler')
    def test_go_download(self, aumock):
        """
            Test that we are actually handing off to charm-helpers to
            download a specific archive of Go. This is non-configurable so
            its reasonably safe to assume we're going to always do this,
            and when it changes we shall curse the brittleness of this test.
        """
        ins_mock = aumock.return_value.install
        install.download_go()
        url = 'https://storage.googleapis.com/golang/go1.4.2.linux-amd64.tar.gz'
        sha1='5020af94b52b65cc9b6f11d50a67e4bae07b0aff'
        ins_mock.assert_called_with(url, '/usr/local', sha1, 'sha1')

    @patch('install.subprocess')
    def test_clone_repository(self, spmock):
        """
         We're not using a unit-tested git library - so ensure our subprocess
         call is consistent. If we change this, we want to know we've broken it.
        """
        install.clone_repository()
        repo = 'https://github.com/GoogleCloudPlatform/kubernetes.git'
        direct = '/opt/kubernetes'
        spmock.check_output.assert_called_with(['git', 'clone', repo, direct])

    @patch('install.install_packages')
    @patch('install.download_go')
    @patch('install.clone_repository')
    @patch('install.update_rc_files')
    @patch('install.hookenv')
    def test_install_main(self, hemock, urmock, crmock, dgmock, ipmock):
        """
        Ensure the driver/main method is calling all the supporting methods.
        """
        strings = [
        'export GOROOT=/usr/local/go\n',
        'export PATH=$PATH:$GOROOT/bin\n',
        'export KUBE_MASTER_IP=0.0.0.0\n',
        'export KUBERNETES_MASTER=http://$KUBE_MASTER_IP\n',
        ]

        install.install()
        crmock.assert_called_once()
        dgmock.assert_called_once()
        crmock.assert_called_once()
        urmock.assert_called_with(strings)
        hemock.open_port.assert_called_with(8080)
