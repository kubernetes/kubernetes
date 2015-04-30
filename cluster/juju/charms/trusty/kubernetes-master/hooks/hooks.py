#!/usr/bin/python
"""
The main hook file is called by Juju.
"""
import contextlib
import os
import socket
import subprocess
import sys
from charmhelpers.core import hookenv, host
from kubernetes_installer import KubernetesInstaller
from path import path

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
    charm_dir = path(hookenv.charm_dir())
    config = hookenv.config()
    # Get the version of kubernetes to install.
    version = config['version']
    # Get the package architecture, rather than the from the kernel (uname -m).
    arch = subprocess.check_output(['dpkg', '--print-architecture']).strip()
    kubernetes_dir = path('/opt/kubernetes')
    if not kubernetes_dir.exists():
        print('The source directory {0} does not exist'.format(kubernetes_dir))
        print('Was the kubernetes code cloned during install?')
        exit(1)

    if version in ['source', 'head', 'master']:
        branch = 'master'
    else:
        # Create a branch to a tag.
        branch = 'tags/{0}'.format(version)

    # Construct the path to the binaries using the arch.
    output_path = kubernetes_dir / '_output/local/bin/linux' / arch
    installer = KubernetesInstaller(arch, version, output_path)

    # Change to the kubernetes directory (git repository).
    with kubernetes_dir:
        # Create a command to get the current branch.
        git_branch = 'git branch | grep "\*" | cut -d" " -f2'
        current_branch = subprocess.check_output(git_branch, shell=True).strip()
        print('Current branch: ', current_branch)
        # Create the path to a file to indicate if the build was broken.
        broken_build = charm_dir / '.broken_build'
        # write out the .broken_build file while this block is executing.
        with check_sentinel(broken_build) as last_build_failed:
            print('Last build failed: ', last_build_failed)
            # Rebuild if the current version is different or last build failed.
            if current_branch != version or last_build_failed:
                installer.build(branch)
        if not output_path.exists():
            broken_build.touch()
        else:
            print('Notifying minions of verison ' + version)
            # Notify the minions of a version change.
            for r in hookenv.relation_ids('minions-api'):
                hookenv.relation_set(r, version=version)
            print('Done notifing minions of version ' + version)

    # Create the symoblic links to the right directories.
    installer.install()

    relation_changed()

    hookenv.log('The config-changed hook completed successfully.')


@hooks.hook('etcd-relation-changed', 'minions-api-relation-changed')
def relation_changed():
    template_data = get_template_data()

    # Check required keys
    for k in ('etcd_servers',):
        if not template_data.get(k):
            print "Missing data for", k, template_data
            return

    print "Running with\n", template_data

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


def notify_minions():
    print("Notify minions.")
    config = hookenv.config()
    for r in hookenv.relation_ids('minions-api'):
        hookenv.relation_set(
            r,
            hostname=hookenv.unit_private_ip(),
            port=8080,
            version=config['version'])


def get_template_data():
    rels = hookenv.relations()
    config = hookenv.config()
    template_data = {}
    template_data['etcd_servers'] = ",".join([
        "http://%s:%s" % (s[0], s[1]) for s in sorted(
            get_rel_hosts('etcd', rels, ('hostname', 'port')))])
    template_data['minions'] = ",".join(get_rel_hosts('minions-api', rels))

    template_data['api_bind_address'] = _bind_addr(hookenv.unit_private_ip())
    template_data['bind_address'] = "127.0.0.1"
    template_data['api_server_address'] = "http://%s:%s" % (
        hookenv.unit_private_ip(), 8080)
    arch = subprocess.check_output(['dpkg', '--print-architecture']).strip()
    template_data['web_uri'] = "/kubernetes/%s/local/bin/linux/%s/" % (
        config['version'], arch)
    _encode(template_data)
    return template_data


def _bind_addr(addr):
    if addr.replace('.', '').isdigit():
        return addr
    try:
        return socket.gethostbyname(addr)
    except socket.error:
            raise ValueError("Could not resolve private address")


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


def render_file(name, data, src_suffix="upstart.tmpl", tgt_path=None):
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
