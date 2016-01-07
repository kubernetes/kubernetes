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
The main hook file is called by Juju.
"""
import contextlib
import os
import socket
import subprocess
import sys
from charmhelpers.core import hookenv, host
from charmhelpers.contrib import ssl
from kubernetes_installer import KubernetesInstaller
from path import Path

hooks = hookenv.Hooks()


@contextlib.contextmanager
def check_sentinel(filepath):
    """
    A context manager method to write a file while the code block is doing
    something and remove the file when done.
    """
    fail = False
    try:
        yield filepath.exists()
    except:
        fail = True
        filepath.touch()
        raise
    finally:
        if fail is False and filepath.exists():
            filepath.remove()


@hooks.hook('config-changed')
def config_changed():
    """
    On the execution of the juju event 'config-changed' this function
    determines the appropriate architecture and the configured version to
    create kubernetes binary files.
    """
    hookenv.log('Starting config-changed')
    charm_dir = Path(hookenv.charm_dir())
    config = hookenv.config()
    # Get the version of kubernetes to install.
    version = config['version']
    username = config['username']
    password = config['password']
    certificate = config['apiserver-cert']
    key = config['apiserver-key']

    if version == 'master':
        # The 'master' branch of kuberentes is used when master is configured.
        branch = 'master'
    elif version == 'local':
        # Check for kubernetes binaries in the local files/output directory.
        branch = None
    else:
        # Create a branch to a tag to get the release version.
        branch = 'tags/{0}'.format(version)

    cert_file = '/srv/kubernetes/apiserver.crt'
    key_file = '/srv/kubernetes/apiserver.key'
    # When the cert or key changes we need to restart the apiserver.
    if config.changed('apiserver-cert') or config.changed('apiserver-key'):
        hookenv.log('Certificate or key has changed.')
        if not certificate or not key:
            generate_cert(key=key_file, cert=cert_file)
        else:
            hookenv.log('Writing new certificate and key to server.')
            with open(key_file, 'w') as file:
                file.write(key)
            with open(cert_file, 'w') as file:
                file.write(certificate)
        # Restart apiserver as the certificate or key has changed.
        if host.service_running('apiserver'):
            host.service_restart('apiserver')
        # Reload nginx because it proxies https to apiserver.
        if host.service_running('nginx'):
            host.service_reload('nginx')

    if config.changed('username') or config.changed('password'):
        hookenv.log('Username or password changed, creating authentication.')
        basic_auth(username, username, password)
        if host.service_running('apiserver'):
            host.service_restart('apiserver')

    # Get package architecture, rather than arch from the kernel (uname -m).
    arch = subprocess.check_output(['dpkg', '--print-architecture']).strip()

    if not branch:
        output_path = charm_dir / 'files/output'
        kube_installer = KubernetesInstaller(arch, version, output_path)
    else:

        # Build the kuberentes binaries from source on the units.
        kubernetes_dir = Path('/opt/kubernetes')

        # Construct the path to the binaries using the arch.
        output_path = kubernetes_dir / '_output/local/bin/linux' / arch
        kube_installer = KubernetesInstaller(arch, version, output_path)

        if not kubernetes_dir.exists():
            message = 'The kubernetes source directory {0} does not exist. ' \
                'Was the kubernetes repository cloned during the install?'
            print(message.format(kubernetes_dir))
            exit(1)

        # Change to the kubernetes directory (git repository).
        with kubernetes_dir:
            # Create a command to get the current branch.
            git_branch = 'git branch | grep "\*" | cut -d" " -f2'
            current_branch = subprocess.check_output(git_branch, shell=True)
            current_branch = current_branch.strip()
            print('Current branch: ', current_branch)
            # Create the path to a file to indicate if the build was broken.
            broken_build = charm_dir / '.broken_build'
            # write out the .broken_build file while this block is executing.
            with check_sentinel(broken_build) as last_build_failed:
                print('Last build failed: ', last_build_failed)
                # Rebuild if current version is different or last build failed.
                if current_branch != version or last_build_failed:
                    kube_installer.build(branch)
            if not output_path.isdir():
                broken_build.touch()

    # Create the symoblic links to the right directories.
    kube_installer.install()

    relation_changed()

    hookenv.log('The config-changed hook completed successfully.')


@hooks.hook('etcd-relation-changed', 'minions-api-relation-changed')
def relation_changed():
    template_data = get_template_data()

    # Check required keys
    for k in ('etcd_servers',):
        if not template_data.get(k):
            print 'Missing data for', k, template_data
            return

    print 'Running with\n', template_data

    # Render and restart as needed
    for n in ('apiserver', 'controller-manager', 'scheduler'):
        if render_file(n, template_data) or not host.service_running(n):
            host.service_restart(n)

    # Render the file that makes the kubernetes binaries available to minions.
    if render_file(
            'distribution', template_data,
            'conf.tmpl', '/etc/nginx/sites-enabled/distribution') or \
            not host.service_running('nginx'):
        host.service_reload('nginx')
    # Render the default nginx template.
    if render_file(
            'nginx', template_data,
            'conf.tmpl', '/etc/nginx/sites-enabled/default') or \
            not host.service_running('nginx'):
        host.service_reload('nginx')

    # Send api endpoint to minions
    notify_minions()


@hooks.hook('network-relation-changed')
def network_relation_changed():
    relation_id = hookenv.relation_id()
    hookenv.relation_set(relation_id, ignore_errors=True)


def notify_minions():
    print('Notify minions.')
    config = hookenv.config()
    for r in hookenv.relation_ids('minions-api'):
        hookenv.relation_set(
            r,
            hostname=hookenv.unit_private_ip(),
            port=8080,
            version=config['version'])
    print('Notified minions of version ' + config['version'])


def basic_auth(name, id, pwd=None, file='/srv/kubernetes/basic-auth.csv'):
    """
    Create a basic authentication file for kubernetes. The file is a csv file
    with 3 columns: password, user name, user id. From the Kubernetes docs:
    The basic auth credentials last indefinitely, and the password cannot be
    changed without restarting apiserver.
    """
    if not pwd:
        import random
        import string
        alphanumeric = string.ascii_letters + string.digits
        pwd = ''.join(random.choice(alphanumeric) for _ in range(16))
    lines = []
    auth_file = Path(file)
    if auth_file.isfile():
        lines = auth_file.lines()
        for line in lines:
            target = ',{0},{1}'.format(name, id)
            if target in line:
                lines.remove(line)
    auth_line = '{0},{1},{2}'.format(pwd, name, id)
    lines.append(auth_line)
    auth_file.write_lines(lines)


def generate_cert(common_name=None,
                  key='/srv/kubernetes/apiserver.key',
                  cert='/srv/kubernetes/apiserver.crt'):
    """
    Create the certificate and key for the Kubernetes tls enablement.
    """
    hookenv.log('Generating new self signed certificate and key', 'INFO')
    if not common_name:
        common_name = hookenv.unit_get('public-address')
    if os.path.isfile(key) or os.path.isfile(cert):
        hookenv.log('Overwriting the existing certificate or key', 'WARNING')
    hookenv.log('Generating certificate for {0}'.format(common_name), 'INFO')
    # Generate the self signed certificate with the public address as CN.
    # https://pythonhosted.org/charmhelpers/api/charmhelpers.contrib.ssl.html
    ssl.generate_selfsigned(key, cert, cn=common_name)


def get_template_data():
    rels = hookenv.relations()
    config = hookenv.config()
    version = config['version']
    template_data = {}
    template_data['etcd_servers'] = ','.join([
        'http://%s:%s' % (s[0], s[1]) for s in sorted(
            get_rel_hosts('etcd', rels, ('hostname', 'port')))])
    template_data['minions'] = ','.join(get_rel_hosts('minions-api', rels))
    private_ip = hookenv.unit_private_ip()
    public_ip = hookenv.unit_public_ip()
    template_data['api_public_address'] = _bind_addr(public_ip)
    template_data['api_private_address'] = _bind_addr(private_ip)
    template_data['bind_address'] = '127.0.0.1'
    template_data['api_http_uri'] = 'http://%s:%s' % (private_ip, 8080)
    template_data['api_https_uri'] = 'https://%s:%s' % (private_ip, 6443)

    arch = subprocess.check_output(['dpkg', '--print-architecture']).strip()

    template_data['web_uri'] = '/kubernetes/%s/local/bin/linux/%s/' % (version,
                                                                       arch)
    if version == 'local':
        template_data['alias'] = hookenv.charm_dir() + '/files/output/'
    else:
        directory = '/opt/kubernetes/_output/local/bin/linux/%s/' % arch
        template_data['alias'] = directory
    _encode(template_data)
    return template_data


def _bind_addr(addr):
    if addr.replace('.', '').isdigit():
        return addr
    try:
        return socket.gethostbyname(addr)
    except socket.error:
            raise ValueError('Could not resolve address %s' % addr)


def _encode(d):
    for k, v in d.items():
        if isinstance(v, unicode):
            d[k] = v.encode('utf8')


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


def render_file(name, data, src_suffix='upstart.tmpl', tgt_path=None):
    tmpl_path = os.path.join(
        os.environ.get('CHARM_DIR'), 'files', '%s.%s' % (name, src_suffix))

    with open(tmpl_path) as fh:
        tmpl = fh.read()
    rendered = tmpl % data

    if tgt_path is None:
        tgt_path = '/etc/init/%s.conf' % name

    if os.path.exists(tgt_path):
        with open(tgt_path) as fh:
            contents = fh.read()
        if contents == rendered:
            return False

    with open(tgt_path, 'w') as fh:
        fh.write(rendered)
    return True


if __name__ == '__main__':
    hooks.execute(sys.argv)
