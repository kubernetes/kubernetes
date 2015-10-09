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

from mock import patch
from mock import ANY
from path import Path
import pytest
import subprocess
import sys

# Add the hooks directory to the python path.
hooks_dir = Path('__file__').parent.abspath() / 'hooks'
sys.path.insert(0, hooks_dir.abspath())
# Import the module to be tested.
import kubernetes_installer


def test_run():
    """ Test the run method both with valid commands and invalid commands. """
    ls = 'ls -l {0}/kubernetes_installer.py'.format(hooks_dir)
    output = kubernetes_installer.run(ls, False)
    assert output
    assert 'kubernetes_installer.py' in output
    output = kubernetes_installer.run(ls, True)
    assert output
    assert 'kubernetes_installer.py' in output

    invalid_directory = Path('/not/a/real/directory')
    assert not invalid_directory.exists()
    invalid_command = 'ls {0}'.format(invalid_directory)
    with pytest.raises(subprocess.CalledProcessError) as error:
        kubernetes_installer.run(invalid_command)
        print(error)
    with pytest.raises(subprocess.CalledProcessError) as error:
        kubernetes_installer.run(invalid_command, shell=True)
        print(error)


class TestKubernetesInstaller():

    def makeone(self, *args, **kw):
        """ Create the KubernetesInstaller object and return it. """
        from kubernetes_installer import KubernetesInstaller
        return KubernetesInstaller(*args, **kw)

    def test_init(self):
        """ Test that the init method correctly assigns the variables. """
        ki = self.makeone('i386', '3.0.1', '/tmp/does_not_exist')
        assert ki.aliases
        assert 'kube-apiserver' in ki.aliases
        assert 'kube-controller-manager' in ki.aliases
        assert 'kube-scheduler' in ki.aliases
        assert 'kubectl' in ki.aliases
        assert 'kubelet' in ki.aliases
        assert ki.arch == 'i386'
        assert ki.version == '3.0.1'
        assert ki.output_dir == Path('/tmp/does_not_exist')

    @patch('kubernetes_installer.run')
    @patch('kubernetes_installer.subprocess.call')
    def test_build(self, cmock, rmock):
        """ Test the build method with master and non-master branches. """
        directory = Path('/tmp/kubernetes_installer_test/build')
        ki = self.makeone('amd64', 'v99.00.11', directory)
        assert not directory.exists(), 'The %s directory exists!' % directory
        # Call the build method with "master" branch.
        ki.build("master")
        # TODO: run is called many times but mock only remembers last one.
        rmock.assert_called_with('git reset --hard origin/master')

        # TODO: call is complex and hard to verify with mock, fix that.
        # this is not doing what we think it should be doing, magic mock
        # makes this tricky.
        # list['foo', 'baz'], env = ANY
        make_args = ['make', 'all', 'WHAT=cmd/kube-apiserver cmd/kubectl cmd/kube-controller-manager plugin/cmd/kube-scheduler cmd/kubelet cmd/kube-proxy']  # noqa
        cmock.assert_called_once_with(make_args, env=ANY)

    @patch('kubernetes_installer.run')
    @patch('kubernetes_installer.subprocess.call')
    def test_schenanigans(self, cmock, rmock):
        """ Test the build method with master and non-master branches. """
        directory = Path('/tmp/kubernetes_installer_test/build')
        ki = self.makeone('amd64', 'v99.00.11', directory)
        assert not directory.exists(), 'The %s directory exists!' % directory

        # Call the build method with something other than "master" branch.
        ki.build("branch")
        # TODO: run is called many times, but mock only remembers last one.
        rmock.assert_called_with('git checkout -b v99.00.11 branch')
        # TODO: call is complex and hard to verify with mock, fix that.
        assert cmock.called

        directory.rmtree_p()

    def test_install(self):
        """ Test the install method that it creates the correct links. """
        directory = Path('/tmp/kubernetes_installer_test/install')
        ki = self.makeone('ppc64le', '1.2.3', directory)
        assert not directory.exists(), 'The %s directory exits!' % directory
        directory.makedirs_p()
        # Create the files for the install method to link to.
        (directory / 'kube-apiserver').touch()
        (directory / 'kube-controller-manager').touch()
        (directory / 'kube-proxy').touch()
        (directory / 'kube-scheduler').touch()
        (directory / 'kubectl').touch()
        (directory / 'kubelet').touch()

        results = directory / 'install/results/go/here'
        assert not results.exists()
        ki.install(results)
        assert results.isdir()
        # Check that all the files were correctly aliased and are links.
        assert (results / 'apiserver').islink()
        assert (results / 'controller-manager').islink()
        assert (results / 'kube-proxy').islink()
        assert (results / 'scheduler').islink()
        assert (results / 'kubectl').islink()
        assert (results / 'kubelet').islink()

        directory.rmtree_p()
