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

"""
The main hook file that is called by Juju.
"""
import os
import socket
import subprocess
import sys
import urlparse

from charmhelpers.core import hookenv, host
from kubernetes_installer import KubernetesInstaller
from path import Path

from lib.registrator import Registrator

hooks = hookenv.Hooks()


@hooks.hook('api-relation-changed')
def api_relation_changed():
    """
    On the relation to the api server, this function determines the appropriate
    architecture and the configured version to copy the kubernetes binary files
    from the kubernetes-master charm and installs it locally on this machine.
    """
    hookenv.log('Starting api-relation-changed')
    charm_dir = Path(hookenv.charm_dir())
    # Get the package architecture, rather than the from the kernel (uname -m).
    arch = subprocess.check_output(['dpkg', '--print-architecture']).strip()
    kubernetes_bin_dir = Path('/opt/kubernetes/bin')
    # Get the version of kubernetes to install.
    version = subprocess.check_output(['relation-get', 'version']).strip()
    print('Relation version: ', version)
    if not version:
        print('No version present in the relation.')
        exit(0)
    version_file = charm_dir / '.version'
    if version_file.exists():
        previous_version = version_file.text()
        print('Previous version: ', previous_version)
        if version == previous_version:
            exit(0)
    # Can not download binaries while the service is running, so stop it.
    # TODO: Figure out a better way to handle upgraded kubernetes binaries.
    for service in ('kubelet', 'proxy'):
        if host.service_running(service):
            host.service_stop(service)
    command = ['relation-get', 'private-address']
    # Get the kubernetes-master address.
    server = subprocess.check_output(command).strip()
    print('Kubernetes master private address: ', server)
    installer = KubernetesInstaller(arch, version, server, kubernetes_bin_dir)
    installer.download()
    installer.install()
    # Write the most recently installed version number to the file.
    version_file.write_text(version)
    relation_changed()


@hooks.hook('etcd-relation-changed',
            'network-relation-changed')
def relation_changed():
    """Connect the parts and go :-)
    """
    template_data = get_template_data()

    # Check required keys
    for k in ('etcd_servers', 'kubeapi_server'):
        if not template_data.get(k):
            print('Missing data for %s %s' % (k, template_data))
            return
    print('Running with\n%s' % template_data)

    # Setup kubernetes supplemental group
    setup_kubernetes_group()

    # Register upstart managed services
    for n in ('kubelet', 'proxy'):
        if render_upstart(n, template_data) or not host.service_running(n):
            print('Starting %s' % n)
            host.service_restart(n)

    # Register machine via api
    print('Registering machine')
    register_machine(template_data['kubeapi_server'])

    # Save the marker (for restarts to detect prev install)
    template_data.save()


def get_template_data():
    rels = hookenv.relations()
    template_data = hookenv.Config()
    template_data.CONFIG_FILE_NAME = '.unit-state'

    overlay_type = get_scoped_rel_attr('network', rels, 'overlay_type')
    etcd_servers = get_rel_hosts('etcd', rels, ('hostname', 'port'))
    api_servers = get_rel_hosts('api', rels, ('hostname', 'port'))

    # kubernetes master isn't ha yet.
    if api_servers:
        api_info = api_servers.pop()
        api_servers = 'http://%s:%s' % (api_info[0], api_info[1])

    template_data['overlay_type'] = overlay_type
    template_data['kubelet_bind_addr'] = _bind_addr(
        hookenv.unit_private_ip())
    template_data['proxy_bind_addr'] = _bind_addr(
        hookenv.unit_get('public-address'))
    template_data['kubeapi_server'] = api_servers
    template_data['etcd_servers'] = ','.join([
        'http://%s:%s' % (s[0], s[1]) for s in sorted(etcd_servers)])
    template_data['identifier'] = os.environ['JUJU_UNIT_NAME'].replace(
        '/', '-')
    return _encode(template_data)


def _bind_addr(addr):
    if addr.replace('.', '').isdigit():
        return addr
    try:
        return socket.gethostbyname(addr)
    except socket.error:
            raise ValueError('Could not resolve private address')


def _encode(d):
    for k, v in d.items():
        if isinstance(v, unicode):
            d[k] = v.encode('utf8')
    return d


def get_scoped_rel_attr(rel_name, rels, attr):
    private_ip = hookenv.unit_private_ip()
    for r, data in rels.get(rel_name, {}).items():
        for unit_id, unit_data in data.items():
            if unit_data.get('private-address') != private_ip:
                continue
            if unit_data.get(attr):
                return unit_data.get(attr)


def get_rel_hosts(rel_name, rels, keys=('private-address',)):
    hosts = []
    for r, data in rels.get(rel_name, {}).items():
        for unit_id, unit_data in data.items():
            if unit_id == hookenv.local_unit():
                continue
            values = [unit_data.get(k) for k in keys]
            if not all(values):
                continue
            hosts.append(len(values) == 1 and values[0] or values)
    return hosts


def render_upstart(name, data):
    tmpl_path = os.path.join(
        os.environ.get('CHARM_DIR'), 'files', '%s.upstart.tmpl' % name)

    with open(tmpl_path) as fh:
        tmpl = fh.read()
    rendered = tmpl % data

    tgt_path = '/etc/init/%s.conf' % name

    if os.path.exists(tgt_path):
        with open(tgt_path) as fh:
            contents = fh.read()
        if contents == rendered:
            return False

    with open(tgt_path, 'w') as fh:
        fh.write(rendered)
    return True


def register_machine(apiserver, retry=False):
    parsed = urlparse.urlparse(apiserver)
    # identity = hookenv.local_unit().replace('/', '-')
    private_address = hookenv.unit_private_ip()

    with open('/proc/meminfo') as fh:
        info = fh.readline()
        mem = info.strip().split(':')[1].strip().split()[0]
    cpus = os.sysconf('SC_NPROCESSORS_ONLN')

    # https://github.com/kubernetes/kubernetes/blob/master/docs/admin/node.md
    registration_request = Registrator()
    registration_request.data['kind'] = 'Node'
    registration_request.data['id'] = private_address
    registration_request.data['name'] = private_address
    registration_request.data['metadata']['name'] = private_address
    registration_request.data['spec']['capacity']['mem'] = mem + ' K'
    registration_request.data['spec']['capacity']['cpu'] = cpus
    registration_request.data['spec']['externalID'] = private_address
    registration_request.data['status']['hostIP'] = private_address

    try:
        response, result = registration_request.register(parsed.hostname,
                                                         parsed.port,
                                                         '/api/v1/nodes')
    except socket.error:
        hookenv.status_set('blocked',
                           'Error communicating with Kubenetes Master')
        return

    print(response)

    try:
        registration_request.command_succeeded(response, result)
    except ValueError:
        # This happens when we have already registered
        # for now this is OK
        pass


def setup_kubernetes_group():
    output = subprocess.check_output(['groups', 'kubernetes'])

    # TODO: check group exists
    if 'docker' not in output:
        subprocess.check_output(
            ['usermod', '-a', '-G', 'docker', 'kubernetes'])


if __name__ == '__main__':
    hooks.execute(sys.argv)
