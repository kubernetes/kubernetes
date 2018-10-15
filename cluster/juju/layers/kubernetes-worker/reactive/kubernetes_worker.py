#!/usr/bin/env python

# Copyright 2015 The Kubernetes Authors.
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

import hashlib
import json
import os
import random
import shutil
import subprocess
import time
import yaml

from charms.leadership import leader_get, leader_set

from pathlib import Path
from shlex import split
from subprocess import check_call, check_output
from subprocess import CalledProcessError
from socket import gethostname, getfqdn

from charms import layer
from charms.layer import snap
from charms.reactive import hook
from charms.reactive import endpoint_from_flag
from charms.reactive import set_state, remove_state, is_state
from charms.reactive import when, when_any, when_not, when_none

from charms.kubernetes.common import get_version

from charms.reactive.helpers import data_changed
from charms.templating.jinja2 import render

from charmhelpers.core import hookenv, unitdata
from charmhelpers.core.host import service_stop, service_restart
from charmhelpers.contrib.charmsupport import nrpe

# Override the default nagios shortname regex to allow periods, which we
# need because our bin names contain them (e.g. 'snap.foo.daemon'). The
# default regex in charmhelpers doesn't allow periods, but nagios itself does.
nrpe.Check.shortname_re = '[\.A-Za-z0-9-_]+$'

kubeconfig_path = '/root/cdk/kubeconfig'
kubeproxyconfig_path = '/root/cdk/kubeproxyconfig'
kubeclientconfig_path = '/root/.kube/config'
gcp_creds_env_key = 'GOOGLE_APPLICATION_CREDENTIALS'
snap_resources = ['kubectl', 'kubelet', 'kube-proxy']

os.environ['PATH'] += os.pathsep + os.path.join(os.sep, 'snap', 'bin')
db = unitdata.kv()


@hook('upgrade-charm')
def upgrade_charm():
    # migrate to new flags
    if is_state('kubernetes-worker.restarted-for-cloud'):
        remove_state('kubernetes-worker.restarted-for-cloud')
        set_state('kubernetes-worker.cloud.ready')
    if is_state('kubernetes-worker.cloud-request-sent'):
        # minor change, just for consistency
        remove_state('kubernetes-worker.cloud-request-sent')
        set_state('kubernetes-worker.cloud.request-sent')

    # Trigger removal of PPA docker installation if it was previously set.
    set_state('config.changed.install_from_upstream')
    hookenv.atexit(remove_state, 'config.changed.install_from_upstream')

    cleanup_pre_snap_services()
    migrate_resource_checksums()
    check_resources_for_upgrade_needed()

    # Remove the RC for nginx ingress if it exists
    if hookenv.config().get('ingress'):
        kubectl_success('delete', 'rc', 'nginx-ingress-controller')

    # Remove gpu.enabled state so we can reconfigure gpu-related kubelet flags,
    # since they can differ between k8s versions
    if is_state('kubernetes-worker.gpu.enabled'):
        remove_state('kubernetes-worker.gpu.enabled')
        try:
            disable_gpu()
        except ApplyNodeLabelFailed:
            # Removing node label failed. Probably the master is unavailable.
            # Proceed with the upgrade in hope GPUs will still be there.
            hookenv.log('Failed to remove GPU labels. Proceed with upgrade.')

    remove_state('kubernetes-worker.cni-plugins.installed')
    remove_state('kubernetes-worker.config.created')
    remove_state('kubernetes-worker.ingress.available')
    remove_state('worker.auth.bootstrapped')
    set_state('kubernetes-worker.restart-needed')


def get_resource_checksum_db_key(resource):
    ''' Convert a resource name to a resource checksum database key. '''
    return 'kubernetes-worker.resource-checksums.' + resource


def calculate_resource_checksum(resource):
    ''' Calculate a checksum for a resource '''
    md5 = hashlib.md5()
    path = hookenv.resource_get(resource)
    if path:
        with open(path, 'rb') as f:
            data = f.read()
        md5.update(data)
    return md5.hexdigest()


def migrate_resource_checksums():
    ''' Migrate resource checksums from the old schema to the new one '''
    for resource in snap_resources:
        new_key = get_resource_checksum_db_key(resource)
        if not db.get(new_key):
            path = hookenv.resource_get(resource)
            if path:
                # old key from charms.reactive.helpers.any_file_changed
                old_key = 'reactive.files_changed.' + path
                old_checksum = db.get(old_key)
                db.set(new_key, old_checksum)
            else:
                # No resource is attached. Previously, this meant no checksum
                # would be calculated and stored. But now we calculate it as if
                # it is a 0-byte resource, so let's go ahead and do that.
                zero_checksum = hashlib.md5().hexdigest()
                db.set(new_key, zero_checksum)


def check_resources_for_upgrade_needed():
    hookenv.status_set('maintenance', 'Checking resources')
    for resource in snap_resources:
        key = get_resource_checksum_db_key(resource)
        old_checksum = db.get(key)
        new_checksum = calculate_resource_checksum(resource)
        if new_checksum != old_checksum:
            set_upgrade_needed()


def calculate_and_store_resource_checksums():
    for resource in snap_resources:
        key = get_resource_checksum_db_key(resource)
        checksum = calculate_resource_checksum(resource)
        db.set(key, checksum)


def set_upgrade_needed():
    set_state('kubernetes-worker.snaps.upgrade-needed')
    config = hookenv.config()
    previous_channel = config.previous('channel')
    require_manual = config.get('require-manual-upgrade')
    if previous_channel is None or not require_manual:
        set_state('kubernetes-worker.snaps.upgrade-specified')


def cleanup_pre_snap_services():
    # remove old states
    remove_state('kubernetes-worker.components.installed')

    # disable old services
    services = ['kubelet', 'kube-proxy']
    for service in services:
        hookenv.log('Stopping {0} service.'.format(service))
        service_stop(service)

    # cleanup old files
    files = [
        "/lib/systemd/system/kubelet.service",
        "/lib/systemd/system/kube-proxy.service",
        "/etc/default/kube-default",
        "/etc/default/kubelet",
        "/etc/default/kube-proxy",
        "/srv/kubernetes",
        "/usr/local/bin/kubectl",
        "/usr/local/bin/kubelet",
        "/usr/local/bin/kube-proxy",
        "/etc/kubernetes"
    ]
    for file in files:
        if os.path.isdir(file):
            hookenv.log("Removing directory: " + file)
            shutil.rmtree(file)
        elif os.path.isfile(file):
            hookenv.log("Removing file: " + file)
            os.remove(file)


@when('config.changed.channel')
def channel_changed():
    set_upgrade_needed()


@when('kubernetes-worker.snaps.upgrade-specified')
def install_snaps():
    channel = hookenv.config('channel')
    hookenv.status_set('maintenance', 'Installing kubectl snap')
    snap.install('kubectl', channel=channel, classic=True)
    hookenv.status_set('maintenance', 'Installing kubelet snap')
    snap.install('kubelet', channel=channel, classic=True)
    hookenv.status_set('maintenance', 'Installing kube-proxy snap')
    snap.install('kube-proxy', channel=channel, classic=True)
    calculate_and_store_resource_checksums()
    set_state('kubernetes-worker.snaps.installed')
    set_state('kubernetes-worker.restart-needed')
    remove_state('kubernetes-worker.snaps.upgrade-needed')
    remove_state('kubernetes-worker.snaps.upgrade-specified')


@hook('stop')
def shutdown():
    ''' When this unit is destroyed:
        - delete the current node
        - stop the worker services
    '''
    try:
        if os.path.isfile(kubeconfig_path):
            kubectl('delete', 'node', get_node_name())
    except CalledProcessError:
        hookenv.log('Failed to unregister node.')
    service_stop('snap.kubelet.daemon')
    service_stop('snap.kube-proxy.daemon')


@when('docker.available')
@when_not('kubernetes-worker.cni-plugins.installed')
def install_cni_plugins():
    ''' Unpack the cni-plugins resource '''
    charm_dir = os.getenv('CHARM_DIR')

    # Get the resource via resource_get
    try:
        resource_name = 'cni-{}'.format(arch())
        archive = hookenv.resource_get(resource_name)
    except Exception:
        message = 'Error fetching the cni resource.'
        hookenv.log(message)
        hookenv.status_set('blocked', message)
        return

    if not archive:
        hookenv.log('Missing cni resource.')
        hookenv.status_set('blocked', 'Missing cni resource.')
        return

    # Handle null resource publication, we check if filesize < 1mb
    filesize = os.stat(archive).st_size
    if filesize < 1000000:
        hookenv.status_set('blocked', 'Incomplete cni resource.')
        return

    hookenv.status_set('maintenance', 'Unpacking cni resource.')

    unpack_path = '{}/files/cni'.format(charm_dir)
    os.makedirs(unpack_path, exist_ok=True)
    cmd = ['tar', 'xfvz', archive, '-C', unpack_path]
    hookenv.log(cmd)
    check_call(cmd)

    apps = [
        {'name': 'loopback', 'path': '/opt/cni/bin'}
    ]

    for app in apps:
        unpacked = '{}/{}'.format(unpack_path, app['name'])
        app_path = os.path.join(app['path'], app['name'])
        install = ['install', '-v', '-D', unpacked, app_path]
        hookenv.log(install)
        check_call(install)

    # Used by the "registry" action. The action is run on a single worker, but
    # the registry pod can end up on any worker, so we need this directory on
    # all the workers.
    os.makedirs('/srv/registry', exist_ok=True)

    set_state('kubernetes-worker.cni-plugins.installed')


@when('kubernetes-worker.snaps.installed')
def set_app_version():
    ''' Declare the application version to juju '''
    cmd = ['kubelet', '--version']
    version = check_output(cmd)
    hookenv.application_version_set(version.split(b' v')[-1].rstrip())


@when('kubernetes-worker.snaps.installed')
@when('snap.refresh.set')
@when('leadership.is_leader')
def process_snapd_timer():
    ''' Set the snapd refresh timer on the leader so all cluster members
    (present and future) will refresh near the same time. '''
    # Get the current snapd refresh timer; we know layer-snap has set this
    # when the 'snap.refresh.set' flag is present.
    timer = snap.get(snapname='core', key='refresh.timer').decode('utf-8')

    # The first time through, data_changed will be true. Subsequent calls
    # should only update leader data if something changed.
    if data_changed('worker_snapd_refresh', timer):
        hookenv.log('setting snapd_refresh timer to: {}'.format(timer))
        leader_set({'snapd_refresh': timer})


@when('kubernetes-worker.snaps.installed')
@when('snap.refresh.set')
@when('leadership.changed.snapd_refresh')
@when_not('leadership.is_leader')
def set_snapd_timer():
    ''' Set the snapd refresh.timer on non-leader cluster members. '''
    # NB: This method should only be run when 'snap.refresh.set' is present.
    # Layer-snap will always set a core refresh.timer, which may not be the
    # same as our leader. Gating with 'snap.refresh.set' ensures layer-snap
    # has finished and we are free to set our config to the leader's timer.
    timer = leader_get('snapd_refresh')
    hookenv.log('setting snapd_refresh timer to: {}'.format(timer))
    snap.set_refresh_timer(timer)


@hookenv.atexit
def charm_status():
    '''Update the status message with the current status of kubelet.'''
    vsphere_joined = is_state('endpoint.vsphere.joined')
    azure_joined = is_state('endpoint.azure.joined')
    cloud_blocked = is_state('kubernetes-worker.cloud.blocked')
    if vsphere_joined and cloud_blocked:
        hookenv.status_set('blocked',
                           'vSphere integration requires K8s 1.12 or greater')
        return
    if azure_joined and cloud_blocked:
        hookenv.status_set('blocked',
                           'Azure integration requires K8s 1.11 or greater')
        return
    if is_state('kubernetes-worker.cloud.pending'):
        hookenv.status_set('waiting', 'Waiting for cloud integration')
        return
    if not is_state('kube-control.dns.available'):
        # During deployment the worker has to start kubelet without cluster dns
        # configured. If this is the first unit online in a service pool
        # waiting to self host the dns pod, and configure itself to query the
        # dns service declared in the kube-system namespace
        hookenv.status_set('waiting', 'Waiting for cluster DNS.')
        return
    if is_state('kubernetes-worker.snaps.upgrade-specified'):
        hookenv.status_set('waiting', 'Upgrade pending')
        return
    if is_state('kubernetes-worker.snaps.upgrade-needed'):
        hookenv.status_set('blocked',
                           'Needs manual upgrade, run the upgrade action')
        return
    if is_state('kubernetes-worker.snaps.installed'):
        update_kubelet_status()
        return
    else:
        pass  # will have been set by snap layer or other handler


def update_kubelet_status():
    ''' There are different states that the kubelet can be in, where we are
    waiting for dns, waiting for cluster turnup, or ready to serve
    applications.'''
    services = [
        'kubelet',
        'kube-proxy'
    ]
    failing_services = []
    for service in services:
        daemon = 'snap.{}.daemon'.format(service)
        if not _systemctl_is_active(daemon):
            failing_services.append(service)

    if len(failing_services) == 0:
        hookenv.status_set('active', 'Kubernetes worker running.')
    else:
        msg = 'Waiting for {} to start.'.format(','.join(failing_services))
        hookenv.status_set('waiting', msg)


def get_ingress_address(relation):
    try:
        network_info = hookenv.network_get(relation.relation_name)
    except NotImplementedError:
        network_info = []

    if network_info and 'ingress-addresses' in network_info:
        # just grab the first one for now, maybe be more robust here?
        return network_info['ingress-addresses'][0]
    else:
        # if they don't have ingress-addresses they are running a juju that
        # doesn't support spaces, so just return the private address
        return hookenv.unit_get('private-address')


@when('certificates.available', 'kube-control.connected')
def send_data(tls, kube_control):
    '''Send the data that is required to create a server certificate for
    this server.'''
    # Use the public ip of this unit as the Common Name for the certificate.
    common_name = hookenv.unit_public_ip()

    ingress_ip = get_ingress_address(kube_control)

    # Create SANs that the tls layer will add to the server cert.
    sans = [
        hookenv.unit_public_ip(),
        ingress_ip,
        gethostname()
    ]

    # Create a path safe name by removing path characters from the unit name.
    certificate_name = hookenv.local_unit().replace('/', '_')

    # Request a server cert with this information.
    tls.request_server_cert(common_name, sans, certificate_name)


@when('kube-api-endpoint.available', 'kube-control.dns.available',
      'cni.available')
def watch_for_changes(kube_api, kube_control, cni):
    ''' Watch for configuration changes and signal if we need to restart the
    worker services '''
    servers = get_kube_api_servers(kube_api)
    dns = kube_control.get_dns()
    cluster_cidr = cni.get_config()['cidr']

    if (data_changed('kube-api-servers', servers) or
            data_changed('kube-dns', dns) or
            data_changed('cluster-cidr', cluster_cidr)):

        set_state('kubernetes-worker.restart-needed')


@when('kubernetes-worker.snaps.installed', 'kube-api-endpoint.available',
      'tls_client.ca.saved', 'tls_client.client.certificate.saved',
      'tls_client.client.key.saved', 'tls_client.server.certificate.saved',
      'tls_client.server.key.saved',
      'kube-control.dns.available', 'kube-control.auth.available',
      'cni.available', 'kubernetes-worker.restart-needed',
      'worker.auth.bootstrapped')
@when_not('kubernetes-worker.cloud.pending',
          'kubernetes-worker.cloud.blocked')
def start_worker(kube_api, kube_control, auth_control, cni):
    ''' Start kubelet using the provided API and DNS info.'''
    servers = get_kube_api_servers(kube_api)
    # Note that the DNS server doesn't necessarily exist at this point. We know
    # what its IP will eventually be, though, so we can go ahead and configure
    # kubelet with that info. This ensures that early pods are configured with
    # the correct DNS even though the server isn't ready yet.

    dns = kube_control.get_dns()
    ingress_ip = get_ingress_address(kube_control)
    cluster_cidr = cni.get_config()['cidr']

    if cluster_cidr is None:
        hookenv.log('Waiting for cluster cidr.')
        return

    creds = db.get('credentials')
    data_changed('kube-control.creds', creds)

    create_config(random.choice(servers), creds)
    configure_kubelet(dns, ingress_ip)
    configure_kube_proxy(servers, cluster_cidr)
    set_state('kubernetes-worker.config.created')
    restart_unit_services()
    update_kubelet_status()
    set_state('kubernetes-worker.label-config-required')
    remove_state('kubernetes-worker.restart-needed')


@when('cni.connected')
@when_not('cni.configured')
def configure_cni(cni):
    ''' Set worker configuration on the CNI relation. This lets the CNI
    subordinate know that we're the worker so it can respond accordingly. '''
    cni.set_config(is_master=False, kubeconfig_path=kubeconfig_path)


@when('config.changed.ingress')
def toggle_ingress_state():
    ''' Ingress is a toggled state. Remove ingress.available if set when
    toggled '''
    remove_state('kubernetes-worker.ingress.available')


@when('docker.sdn.configured')
def sdn_changed():
    '''The Software Defined Network changed on the container so restart the
    kubernetes services.'''
    restart_unit_services()
    update_kubelet_status()
    remove_state('docker.sdn.configured')


@when('kubernetes-worker.config.created')
@when_not('kubernetes-worker.ingress.available')
def render_and_launch_ingress():
    ''' If configuration has ingress daemon set enabled, launch the ingress
    load balancer and default http backend. Otherwise attempt deletion. '''
    config = hookenv.config()
    # If ingress is enabled, launch the ingress controller
    if config.get('ingress'):
        launch_default_ingress_controller()
    else:
        hookenv.log('Deleting the http backend and ingress.')
        kubectl_manifest('delete',
                         '/root/cdk/addons/default-http-backend.yaml')
        kubectl_manifest('delete',
                         '/root/cdk/addons/ingress-daemon-set.yaml')  # noqa
        hookenv.close_port(80)
        hookenv.close_port(443)


@when('config.changed.labels')
def handle_labels_changed():
    set_state('kubernetes-worker.label-config-required')


@when('kubernetes-worker.label-config-required',
      'kubernetes-worker.config.created')
def apply_node_labels():
    ''' Parse the labels configuration option and apply the labels to the
        node. '''
    # Get the user's configured labels.
    config = hookenv.config()
    user_labels = {}
    for item in config.get('labels').split(' '):
        if '=' in item:
            key, val = item.split('=')
            user_labels[key] = val
        else:
            hookenv.log('Skipping malformed option: {}.'.format(item))
    # Collect the current label state.
    current_labels = db.get('current_labels') or {}
    # Remove any labels that the user has removed from the config.
    for key in list(current_labels.keys()):
        if key not in user_labels:
            try:
                remove_label(key)
                del current_labels[key]
                db.set('current_labels', current_labels)
            except ApplyNodeLabelFailed as e:
                hookenv.log(str(e))
                return
    # Add any new labels.
    for key, val in user_labels.items():
        try:
            set_label(key, val)
            current_labels[key] = val
            db.set('current_labels', current_labels)
        except ApplyNodeLabelFailed as e:
            hookenv.log(str(e))
            return
    # Set the juju-application label.
    try:
        set_label('juju-application', hookenv.service_name())
    except ApplyNodeLabelFailed as e:
        hookenv.log(str(e))
        return
    # Label configuration complete.
    remove_state('kubernetes-worker.label-config-required')


@when_any('config.changed.kubelet-extra-args',
          'config.changed.proxy-extra-args',
          'config.changed.kubelet-extra-config')
def config_changed_requires_restart():
    set_state('kubernetes-worker.restart-needed')


@when('config.changed.docker-logins')
def docker_logins_changed():
    """Set a flag to handle new docker login options.

    If docker daemon options have also changed, set a flag to ensure the
    daemon is restarted prior to running docker login.
    """
    config = hookenv.config()

    if data_changed('docker-opts', config['docker-opts']):
        hookenv.log('Found new docker daemon options. Requesting a restart.')
        # State will be removed by layer-docker after restart
        set_state('docker.restart')

    set_state('kubernetes-worker.docker-login')


@when('kubernetes-worker.docker-login')
@when_not('docker.restart')
def run_docker_login():
    """Login to a docker registry with configured credentials."""
    config = hookenv.config()

    previous_logins = config.previous('docker-logins')
    logins = config['docker-logins']
    logins = json.loads(logins)

    if previous_logins:
        previous_logins = json.loads(previous_logins)
        next_servers = {login['server'] for login in logins}
        previous_servers = {login['server'] for login in previous_logins}
        servers_to_logout = previous_servers - next_servers
        for server in servers_to_logout:
            cmd = ['docker', 'logout', server]
            subprocess.check_call(cmd)

    for login in logins:
        server = login['server']
        username = login['username']
        password = login['password']
        cmd = ['docker', 'login', server, '-u', username, '-p', password]
        subprocess.check_call(cmd)

    remove_state('kubernetes-worker.docker-login')
    set_state('kubernetes-worker.restart-needed')


def arch():
    '''Return the package architecture as a string. Raise an exception if the
    architecture is not supported by kubernetes.'''
    # Get the package architecture for this system.
    architecture = check_output(['dpkg', '--print-architecture']).rstrip()
    # Convert the binary result into a string.
    architecture = architecture.decode('utf-8')
    return architecture


def create_config(server, creds):
    '''Create a kubernetes configuration for the worker unit.'''
    # Get the options from the tls-client layer.
    layer_options = layer.options('tls-client')
    # Get all the paths to the tls information required for kubeconfig.
    ca = layer_options.get('ca_certificate_path')

    # Create kubernetes configuration in the default location for ubuntu.
    create_kubeconfig('/home/ubuntu/.kube/config', server, ca,
                      token=creds['client_token'], user='ubuntu')
    # Make the config dir readable by the ubuntu users so juju scp works.
    cmd = ['chown', '-R', 'ubuntu:ubuntu', '/home/ubuntu/.kube']
    check_call(cmd)
    # Create kubernetes configuration in the default location for root.
    create_kubeconfig(kubeclientconfig_path, server, ca,
                      token=creds['client_token'], user='root')
    # Create kubernetes configuration for kubelet, and kube-proxy services.
    create_kubeconfig(kubeconfig_path, server, ca,
                      token=creds['kubelet_token'], user='kubelet')
    create_kubeconfig(kubeproxyconfig_path, server, ca,
                      token=creds['proxy_token'], user='kube-proxy')


def parse_extra_args(config_key):
    elements = hookenv.config().get(config_key, '').split()
    args = {}

    for element in elements:
        if '=' in element:
            key, _, value = element.partition('=')
            args[key] = value
        else:
            args[element] = 'true'

    return args


def configure_kubernetes_service(service, base_args, extra_args_key):
    db = unitdata.kv()

    prev_args_key = 'kubernetes-worker.prev_args.' + service
    prev_args = db.get(prev_args_key) or {}

    extra_args = parse_extra_args(extra_args_key)

    args = {}
    for arg in prev_args:
        # remove previous args by setting to null
        args[arg] = 'null'
    for k, v in base_args.items():
        args[k] = v
    for k, v in extra_args.items():
        args[k] = v

    cmd = ['snap', 'set', service] + ['%s=%s' % item for item in args.items()]
    check_call(cmd)

    db.set(prev_args_key, args)


def merge_kubelet_extra_config(config, extra_config):
    ''' Updates config to include the contents of extra_config. This is done
    recursively to allow deeply nested dictionaries to be merged.

    This is destructive: it modifies the config dict that is passed in.
    '''
    for k, extra_config_value in extra_config.items():
        if isinstance(extra_config_value, dict):
            config_value = config.setdefault(k, {})
            merge_kubelet_extra_config(config_value, extra_config_value)
        else:
            config[k] = extra_config_value


def configure_kubelet(dns, ingress_ip):
    layer_options = layer.options('tls-client')
    ca_cert_path = layer_options.get('ca_certificate_path')
    server_cert_path = layer_options.get('server_certificate_path')
    server_key_path = layer_options.get('server_key_path')

    kubelet_opts = {}
    kubelet_opts['require-kubeconfig'] = 'true'
    kubelet_opts['kubeconfig'] = kubeconfig_path
    kubelet_opts['network-plugin'] = 'cni'
    kubelet_opts['v'] = '0'
    kubelet_opts['logtostderr'] = 'true'
    kubelet_opts['node-ip'] = ingress_ip
    kubelet_opts['allow-privileged'] = set_privileged()

    if is_state('endpoint.aws.ready'):
        kubelet_opts['cloud-provider'] = 'aws'
    elif is_state('endpoint.gcp.ready'):
        cloud_config_path = _cloud_config_path('kubelet')
        kubelet_opts['cloud-provider'] = 'gce'
        kubelet_opts['cloud-config'] = str(cloud_config_path)
    elif is_state('endpoint.openstack.ready'):
        cloud_config_path = _cloud_config_path('kubelet')
        kubelet_opts['cloud-provider'] = 'openstack'
        kubelet_opts['cloud-config'] = str(cloud_config_path)
    elif is_state('endpoint.vsphere.joined'):
        # vsphere just needs to be joined on the worker (vs 'ready')
        cloud_config_path = _cloud_config_path('kubelet')
        kubelet_opts['cloud-provider'] = 'vsphere'
        # NB: vsphere maps node product-id to its uuid (no config file needed).
        uuid_file = '/sys/class/dmi/id/product_uuid'
        with open(uuid_file, 'r') as f:
            uuid = f.read().strip()
        kubelet_opts['provider-id'] = 'vsphere://{}'.format(uuid)
    elif is_state('endpoint.azure.ready'):
        azure = endpoint_from_flag('endpoint.azure.ready')
        cloud_config_path = _cloud_config_path('kubelet')
        kubelet_opts['cloud-provider'] = 'azure'
        kubelet_opts['cloud-config'] = str(cloud_config_path)
        kubelet_opts['provider-id'] = azure.vm_id

    if get_version('kubelet') >= (1, 10):
        # Put together the KubeletConfiguration data
        kubelet_config = {
            'apiVersion': 'kubelet.config.k8s.io/v1beta1',
            'kind': 'KubeletConfiguration',
            'address': '0.0.0.0',
            'authentication': {
                'anonymous': {
                    'enabled': False
                },
                'x509': {
                    'clientCAFile': ca_cert_path
                }
            },
            'clusterDomain': dns['domain'],
            'failSwapOn': False,
            'port': 10250,
            'tlsCertFile': server_cert_path,
            'tlsPrivateKeyFile': server_key_path
        }
        if dns['enable-kube-dns']:
            kubelet_config['clusterDNS'] = [dns['sdn-ip']]
        if is_state('kubernetes-worker.gpu.enabled'):
            kubelet_config['featureGates'] = {
                'DevicePlugins': True
            }

        # Add kubelet-extra-config. This needs to happen last so that it
        # overrides any config provided by the charm.
        kubelet_extra_config = hookenv.config('kubelet-extra-config')
        kubelet_extra_config = yaml.load(kubelet_extra_config)
        merge_kubelet_extra_config(kubelet_config, kubelet_extra_config)

        # Render the file and configure Kubelet to use it
        os.makedirs('/root/cdk/kubelet', exist_ok=True)
        with open('/root/cdk/kubelet/config.yaml', 'w') as f:
            f.write('# Generated by kubernetes-worker charm, do not edit\n')
            yaml.dump(kubelet_config, f)
        kubelet_opts['config'] = '/root/cdk/kubelet/config.yaml'
    else:
        # NOTE: This is for 1.9. Once we've dropped 1.9 support, we can remove
        # this whole block and the parent if statement.
        kubelet_opts['address'] = '0.0.0.0'
        kubelet_opts['anonymous-auth'] = 'false'
        kubelet_opts['client-ca-file'] = ca_cert_path
        kubelet_opts['cluster-domain'] = dns['domain']
        kubelet_opts['fail-swap-on'] = 'false'
        kubelet_opts['port'] = '10250'
        kubelet_opts['tls-cert-file'] = server_cert_path
        kubelet_opts['tls-private-key-file'] = server_key_path
        if dns['enable-kube-dns']:
            kubelet_opts['cluster-dns'] = dns['sdn-ip']
        if is_state('kubernetes-worker.gpu.enabled'):
            kubelet_opts['feature-gates'] = 'DevicePlugins=true'

    if get_version('kubelet') >= (1, 11):
        kubelet_opts['dynamic-config-dir'] = '/root/cdk/kubelet/dynamic-config'

    configure_kubernetes_service('kubelet', kubelet_opts, 'kubelet-extra-args')


def configure_kube_proxy(api_servers, cluster_cidr):
    kube_proxy_opts = {}
    kube_proxy_opts['cluster-cidr'] = cluster_cidr
    kube_proxy_opts['kubeconfig'] = kubeproxyconfig_path
    kube_proxy_opts['logtostderr'] = 'true'
    kube_proxy_opts['v'] = '0'
    kube_proxy_opts['master'] = random.choice(api_servers)
    kube_proxy_opts['hostname-override'] = get_node_name()

    if b'lxc' in check_output('virt-what', shell=True):
        kube_proxy_opts['conntrack-max-per-core'] = '0'

    configure_kubernetes_service('kube-proxy', kube_proxy_opts,
                                 'proxy-extra-args')


def create_kubeconfig(kubeconfig, server, ca, key=None, certificate=None,
                      user='ubuntu', context='juju-context',
                      cluster='juju-cluster', password=None, token=None):
    '''Create a configuration for Kubernetes based on path using the supplied
    arguments for values of the Kubernetes server, CA, key, certificate, user
    context and cluster.'''
    if not key and not certificate and not password and not token:
        raise ValueError('Missing authentication mechanism.')

    # token and password are mutually exclusive. Error early if both are
    # present. The developer has requested an impossible situation.
    # see: kubectl config set-credentials --help
    if token and password:
        raise ValueError('Token and Password are mutually exclusive.')
    # Create the config file with the address of the master server.
    cmd = 'kubectl config --kubeconfig={0} set-cluster {1} ' \
          '--server={2} --certificate-authority={3} --embed-certs=true'
    check_call(split(cmd.format(kubeconfig, cluster, server, ca)))
    # Delete old users
    cmd = 'kubectl config --kubeconfig={0} unset users'
    check_call(split(cmd.format(kubeconfig)))
    # Create the credentials using the client flags.
    cmd = 'kubectl config --kubeconfig={0} ' \
          'set-credentials {1} '.format(kubeconfig, user)

    if key and certificate:
        cmd = '{0} --client-key={1} --client-certificate={2} '\
              '--embed-certs=true'.format(cmd, key, certificate)
    if password:
        cmd = "{0} --username={1} --password={2}".format(cmd, user, password)
    # This is mutually exclusive from password. They will not work together.
    if token:
        cmd = "{0} --token={1}".format(cmd, token)
    check_call(split(cmd))
    # Create a default context with the cluster.
    cmd = 'kubectl config --kubeconfig={0} set-context {1} ' \
          '--cluster={2} --user={3}'
    check_call(split(cmd.format(kubeconfig, context, cluster, user)))
    # Make the config use this new context.
    cmd = 'kubectl config --kubeconfig={0} use-context {1}'
    check_call(split(cmd.format(kubeconfig, context)))


@when_any('config.changed.default-backend-image',
          'config.changed.ingress-ssl-chain-completion',
          'config.changed.nginx-image')
@when('kubernetes-worker.config.created')
def launch_default_ingress_controller():
    ''' Launch the Kubernetes ingress controller & default backend (404) '''
    config = hookenv.config()

    # need to test this in case we get in
    # here from a config change to the image
    if not config.get('ingress'):
        return

    context = {}
    context['arch'] = arch()
    addon_path = '/root/cdk/addons/{}'

    context['defaultbackend_image'] = config.get('default-backend-image')
    if (context['defaultbackend_image'] == "" or
       context['defaultbackend_image'] == "auto"):
        if context['arch'] == 's390x':
            context['defaultbackend_image'] = \
                "k8s.gcr.io/defaultbackend-s390x:1.5"
        elif context['arch'] == 'arm64':
            context['defaultbackend_image'] = \
                "k8s.gcr.io/defaultbackend-arm64:1.5"
        else:
            context['defaultbackend_image'] = \
                "k8s.gcr.io/defaultbackend-amd64:1.5"

    # Render the default http backend (404) replicationcontroller manifest
    manifest = addon_path.format('default-http-backend.yaml')
    render('default-http-backend.yaml', manifest, context)
    hookenv.log('Creating the default http backend.')
    try:
        kubectl('apply', '-f', manifest)
    except CalledProcessError as e:
        hookenv.log(e)
        hookenv.log('Failed to create default-http-backend. Will attempt again next update.')  # noqa
        hookenv.close_port(80)
        hookenv.close_port(443)
        return

    # Render the ingress daemon set controller manifest
    context['ssl_chain_completion'] = config.get(
        'ingress-ssl-chain-completion')
    context['ingress_image'] = config.get('nginx-image')
    if context['ingress_image'] == "" or context['ingress_image'] == "auto":
        images = {'amd64': 'quay.io/kubernetes-ingress-controller/nginx-ingress-controller:0.16.1',  # noqa
                  'arm64': 'quay.io/kubernetes-ingress-controller/nginx-ingress-controller-arm64:0.16.1',  # noqa
                  's390x': 'quay.io/kubernetes-ingress-controller/nginx-ingress-controller-s390x:0.16.1',  # noqa
                  'ppc64el': 'quay.io/kubernetes-ingress-controller/nginx-ingress-controller-ppc64le:0.16.1',  # noqa
                  }
        context['ingress_image'] = images.get(context['arch'], images['amd64'])
    if get_version('kubelet') < (1, 9):
        context['daemonset_api_version'] = 'extensions/v1beta1'
    else:
        context['daemonset_api_version'] = 'apps/v1beta2'
    context['juju_application'] = hookenv.service_name()
    manifest = addon_path.format('ingress-daemon-set.yaml')
    render('ingress-daemon-set.yaml', manifest, context)
    hookenv.log('Creating the ingress daemon set.')
    try:
        kubectl('apply', '-f', manifest)
    except CalledProcessError as e:
        hookenv.log(e)
        hookenv.log('Failed to create ingress controller. Will attempt again next update.')  # noqa
        hookenv.close_port(80)
        hookenv.close_port(443)
        return

    set_state('kubernetes-worker.ingress.available')
    hookenv.open_port(80)
    hookenv.open_port(443)


def restart_unit_services():
    '''Restart worker services.'''
    hookenv.log('Restarting kubelet and kube-proxy.')
    services = ['kube-proxy', 'kubelet']
    for service in services:
        service_restart('snap.%s.daemon' % service)


def get_kube_api_servers(kube_api):
    '''Return the kubernetes api server address and port for this
    relationship.'''
    hosts = []
    # Iterate over every service from the relation object.
    for service in kube_api.services():
        for unit in service['hosts']:
            hosts.append('https://{0}:{1}'.format(unit['hostname'],
                                                  unit['port']))
    return hosts


def kubectl(*args):
    ''' Run a kubectl cli command with a config file. Returns stdout and throws
    an error if the command fails. '''
    command = ['kubectl', '--kubeconfig=' + kubeclientconfig_path] + list(args)
    hookenv.log('Executing {}'.format(command))
    return check_output(command)


def kubectl_success(*args):
    ''' Runs kubectl with the given args. Returns True if successful, False if
    not. '''
    try:
        kubectl(*args)
        return True
    except CalledProcessError:
        return False


def kubectl_manifest(operation, manifest):
    ''' Wrap the kubectl creation command when using filepath resources
    :param operation - one of get, create, delete, replace
    :param manifest - filepath to the manifest
     '''
    # Deletions are a special case
    if operation == 'delete':
        # Ensure we immediately remove requested resources with --now
        return kubectl_success(operation, '-f', manifest, '--now')
    else:
        # Guard against an error re-creating the same manifest multiple times
        if operation == 'create':
            # If we already have the definition, its probably safe to assume
            # creation was true.
            if kubectl_success('get', '-f', manifest):
                hookenv.log('Skipping definition for {}'.format(manifest))
                return True
        # Execute the requested command that did not match any of the special
        # cases above
        return kubectl_success(operation, '-f', manifest)


@when('nrpe-external-master.available')
@when_not('nrpe-external-master.initial-config')
def initial_nrpe_config(nagios=None):
    set_state('nrpe-external-master.initial-config')
    update_nrpe_config(nagios)


@when('kubernetes-worker.config.created')
@when('nrpe-external-master.available')
@when_any('config.changed.nagios_context',
          'config.changed.nagios_servicegroups')
def update_nrpe_config(unused=None):
    services = ('snap.kubelet.daemon', 'snap.kube-proxy.daemon')
    hostname = nrpe.get_nagios_hostname()
    current_unit = nrpe.get_nagios_unit_name()
    nrpe_setup = nrpe.NRPE(hostname=hostname)
    nrpe.add_init_service_checks(nrpe_setup, services, current_unit)
    nrpe_setup.write()


@when_not('nrpe-external-master.available')
@when('nrpe-external-master.initial-config')
def remove_nrpe_config(nagios=None):
    remove_state('nrpe-external-master.initial-config')

    # List of systemd services for which the checks will be removed
    services = ('snap.kubelet.daemon', 'snap.kube-proxy.daemon')

    # The current nrpe-external-master interface doesn't handle a lot of logic,
    # use the charm-helpers code for now.
    hostname = nrpe.get_nagios_hostname()
    nrpe_setup = nrpe.NRPE(hostname=hostname)

    for service in services:
        nrpe_setup.remove_check(shortname=service)


def set_privileged():
    """Return 'true' if privileged containers are needed.
    This is when a) the user requested them
                 b) user does not care (auto) and GPUs are available in a pre
                    1.9 era
    """
    privileged = hookenv.config('allow-privileged').lower()
    gpu_needs_privileged = (is_state('kubernetes-worker.gpu.enabled') and
                            get_version('kubelet') < (1, 9))

    if privileged == 'auto':
        privileged = 'true' if gpu_needs_privileged else 'false'

    if privileged == 'false' and gpu_needs_privileged:
        disable_gpu()
        remove_state('kubernetes-worker.gpu.enabled')
        # No need to restart kubernetes (set the restart-needed state)
        # because set-privileged is already in the restart path

    return privileged


@when('config.changed.allow-privileged')
@when('kubernetes-worker.config.created')
def on_config_allow_privileged_change():
    """React to changed 'allow-privileged' config value.

    """
    set_state('kubernetes-worker.restart-needed')
    remove_state('config.changed.allow-privileged')


@when('nvidia-docker.installed')
@when('kubernetes-worker.config.created')
@when_not('kubernetes-worker.gpu.enabled')
def enable_gpu():
    """Enable GPU usage on this node.

    """
    if get_version('kubelet') < (1, 9):
        hookenv.status_set(
            'active',
            'Upgrade to snap channel >= 1.9/stable to enable GPU support.'
        )
        return

    hookenv.log('Enabling gpu mode')
    try:
        # Not sure why this is necessary, but if you don't run this, k8s will
        # think that the node has 0 gpus (as shown by the output of
        # `kubectl get nodes -o yaml`
        check_call(['nvidia-smi'])
    except CalledProcessError as cpe:
        hookenv.log('Unable to communicate with the NVIDIA driver.')
        hookenv.log(cpe)
        return

    set_label('gpu', 'true')
    set_label('cuda', 'true')

    set_state('kubernetes-worker.gpu.enabled')
    set_state('kubernetes-worker.restart-needed')


@when('kubernetes-worker.gpu.enabled')
@when_not('nvidia-docker.installed')
@when_not('kubernetes-worker.restart-needed')
def nvidia_departed():
    """Cuda departed, probably due to the docker layer switching to a
     non nvidia-docker."""
    disable_gpu()
    remove_state('kubernetes-worker.gpu.enabled')
    set_state('kubernetes-worker.restart-needed')


def disable_gpu():
    """Disable GPU usage on this node.

    """
    hookenv.log('Disabling gpu mode')

    # Remove node labels
    remove_label('gpu')
    remove_label('cuda')


@when('kubernetes-worker.gpu.enabled')
@when('kube-control.connected')
def notify_master_gpu_enabled(kube_control):
    """Notify kubernetes-master that we're gpu-enabled.

    """
    kube_control.set_gpu(True)


@when_not('kubernetes-worker.gpu.enabled')
@when('kube-control.connected')
def notify_master_gpu_not_enabled(kube_control):
    """Notify kubernetes-master that we're not gpu-enabled.

    """
    kube_control.set_gpu(False)


@when('kube-control.connected')
def request_kubelet_and_proxy_credentials(kube_control):
    """ Request kubelet node authorization with a well formed kubelet user.
    This also implies that we are requesting kube-proxy auth. """

    # The kube-cotrol interface is created to support RBAC.
    # At this point we might as well do the right thing and return the hostname
    # even if it will only be used when we enable RBAC
    nodeuser = 'system:node:{}'.format(get_node_name().lower())
    kube_control.set_auth_request(nodeuser)


@when('kube-control.connected')
def catch_change_in_creds(kube_control):
    """Request a service restart in case credential updates were detected."""
    nodeuser = 'system:node:{}'.format(get_node_name().lower())
    creds = kube_control.get_auth_credentials(nodeuser)
    if creds and creds['user'] == nodeuser:
        # We need to cache the credentials here because if the
        # master changes (master leader dies and replaced by a new one)
        # the new master will have no recollection of our certs.
        db.set('credentials', creds)
        set_state('worker.auth.bootstrapped')
        if data_changed('kube-control.creds', creds):
            set_state('kubernetes-worker.restart-needed')


@when_not('kube-control.connected')
def missing_kube_control():
    """Inform the operator they need to add the kube-control relation.

    If deploying via bundle this won't happen, but if operator is upgrading a
    a charm in a deployment that pre-dates the kube-control relation, it'll be
    missing.

    """
    try:
        goal_state = hookenv.goal_state()
    except NotImplementedError:
        goal_state = {}

    if 'kube-control' in goal_state.get('relations', {}):
        hookenv.status_set(
            'waiting',
            'Waiting for kubernetes-master to become ready')
    else:
        hookenv.status_set(
            'blocked',
            'Relate {}:kube-control kubernetes-master:kube-control'.format(
                hookenv.service_name()))


@when('docker.ready')
def fix_iptables_for_docker_1_13():
    """ Fix iptables FORWARD policy for Docker >=1.13
    https://github.com/kubernetes/kubernetes/issues/40182
    https://github.com/kubernetes/kubernetes/issues/39823
    """
    cmd = ['iptables', '-w', '300', '-P', 'FORWARD', 'ACCEPT']
    check_call(cmd)


def _systemctl_is_active(application):
    ''' Poll systemctl to determine if the application is running '''
    cmd = ['systemctl', 'is-active', application]
    try:
        raw = check_output(cmd)
        return b'active' in raw
    except Exception:
        return False


def get_node_name():
    kubelet_extra_args = parse_extra_args('kubelet-extra-args')
    cloud_provider = kubelet_extra_args.get('cloud-provider', '')
    if is_state('endpoint.aws.ready'):
        cloud_provider = 'aws'
    elif is_state('endpoint.gcp.ready'):
        cloud_provider = 'gce'
    elif is_state('endpoint.openstack.ready'):
        cloud_provider = 'openstack'
    elif is_state('endpoint.vsphere.ready'):
        cloud_provider = 'vsphere'
    elif is_state('endpoint.azure.ready'):
        cloud_provider = 'azure'
    if cloud_provider == 'aws':
        return getfqdn().lower()
    else:
        return gethostname().lower()


class ApplyNodeLabelFailed(Exception):
    pass


def persistent_call(cmd, retry_message):
    deadline = time.time() + 180
    while time.time() < deadline:
        code = subprocess.call(cmd)
        if code == 0:
            return True
        hookenv.log(retry_message)
        time.sleep(1)
    else:
        return False


def set_label(label, value):
    nodename = get_node_name()
    cmd = 'kubectl --kubeconfig={0} label node {1} {2}={3} --overwrite'
    cmd = cmd.format(kubeconfig_path, nodename, label, value)
    cmd = cmd.split()
    retry = 'Failed to apply label %s=%s. Will retry.' % (label, value)
    if not persistent_call(cmd, retry):
        raise ApplyNodeLabelFailed(retry)


def remove_label(label):
    nodename = get_node_name()
    cmd = 'kubectl --kubeconfig={0} label node {1} {2}-'
    cmd = cmd.format(kubeconfig_path, nodename, label)
    cmd = cmd.split()
    retry = 'Failed to remove label {0}. Will retry.'.format(label)
    if not persistent_call(cmd, retry):
        raise ApplyNodeLabelFailed(retry)


@when_any('endpoint.aws.joined',
          'endpoint.gcp.joined',
          'endpoint.openstack.joined',
          'endpoint.vsphere.joined',
          'endpoint.azure.joined')
@when_not('kubernetes-worker.cloud.ready')
def set_cloud_pending():
    k8s_version = get_version('kubelet')
    k8s_1_11 = k8s_version >= (1, 11)
    k8s_1_12 = k8s_version >= (1, 12)
    vsphere_joined = is_state('endpoint.vsphere.joined')
    azure_joined = is_state('endpoint.azure.joined')
    if (vsphere_joined and not k8s_1_12) or (azure_joined and not k8s_1_11):
        set_state('kubernetes-worker.cloud.blocked')
    else:
        remove_state('kubernetes-worker.cloud.blocked')
    set_state('kubernetes-worker.cloud.pending')


@when_any('endpoint.aws.joined',
          'endpoint.gcp.joined',
          'endpoint.azure.joined')
@when('kube-control.cluster_tag.available')
@when_not('kubernetes-worker.cloud.request-sent')
def request_integration():
    hookenv.status_set('maintenance', 'requesting cloud integration')
    kube_control = endpoint_from_flag('kube-control.cluster_tag.available')
    cluster_tag = kube_control.get_cluster_tag()
    if is_state('endpoint.aws.joined'):
        cloud = endpoint_from_flag('endpoint.aws.joined')
        cloud.tag_instance({
            'kubernetes.io/cluster/{}'.format(cluster_tag): 'owned',
        })
        cloud.tag_instance_security_group({
            'kubernetes.io/cluster/{}'.format(cluster_tag): 'owned',
        })
        cloud.tag_instance_subnet({
            'kubernetes.io/cluster/{}'.format(cluster_tag): 'owned',
        })
        cloud.enable_object_storage_management(['kubernetes-*'])
    elif is_state('endpoint.gcp.joined'):
        cloud = endpoint_from_flag('endpoint.gcp.joined')
        cloud.label_instance({
            'k8s-io-cluster-name': cluster_tag,
        })
        cloud.enable_object_storage_management()
    elif is_state('endpoint.azure.joined'):
        cloud = endpoint_from_flag('endpoint.azure.joined')
        cloud.tag_instance({
            'k8s-io-cluster-name': cluster_tag,
        })
        cloud.enable_object_storage_management()
    cloud.enable_instance_inspection()
    cloud.enable_dns_management()
    set_state('kubernetes-worker.cloud.request-sent')
    hookenv.status_set('waiting', 'Waiting for cloud integration')


@when_none('endpoint.aws.joined',
           'endpoint.gcp.joined',
           'endpoint.openstack.joined',
           'endpoint.vsphere.joined',
           'endpoint.azure.joined')
def clear_cloud_flags():
    remove_state('kubernetes-worker.cloud.pending')
    remove_state('kubernetes-worker.cloud.request-sent')
    remove_state('kubernetes-worker.cloud.blocked')
    remove_state('kubernetes-worker.cloud.ready')


@when_any('endpoint.aws.ready',
          'endpoint.gcp.ready',
          'endpoint.openstack.ready',
          'endpoint.vsphere.ready',
          'endpoint.azure.ready')
@when_not('kubernetes-worker.cloud.blocked',
          'kubernetes-worker.cloud.ready')
def cloud_ready():
    remove_state('kubernetes-worker.cloud.pending')
    if is_state('endpoint.gcp.ready'):
        _write_gcp_snap_config('kubelet')
    elif is_state('endpoint.openstack.ready'):
        _write_openstack_snap_config('kubelet')
    elif is_state('endpoint.azure.ready'):
        _write_azure_snap_config('kubelet')
    set_state('kubernetes-worker.cloud.ready')
    set_state('kubernetes-worker.restart-needed')  # force restart


def _snap_common_path(component):
    return Path('/var/snap/{}/common'.format(component))


def _cloud_config_path(component):
    return _snap_common_path(component) / 'cloud-config.conf'


def _gcp_creds_path(component):
    return _snap_common_path(component) / 'gcp-creds.json'


def _daemon_env_path(component):
    return _snap_common_path(component) / 'environment'


def _write_gcp_snap_config(component):
    # gcp requires additional credentials setup
    gcp = endpoint_from_flag('endpoint.gcp.ready')
    creds_path = _gcp_creds_path(component)
    with creds_path.open('w') as fp:
        os.fchmod(fp.fileno(), 0o600)
        fp.write(gcp.credentials)

    # create a cloud-config file that sets token-url to nil to make the
    # services use the creds env var instead of the metadata server, as
    # well as making the cluster multizone
    cloud_config_path = _cloud_config_path(component)
    cloud_config_path.write_text('[Global]\n'
                                 'token-url = nil\n'
                                 'multizone = true\n')

    daemon_env_path = _daemon_env_path(component)
    if daemon_env_path.exists():
        daemon_env = daemon_env_path.read_text()
        if not daemon_env.endswith('\n'):
            daemon_env += '\n'
    else:
        daemon_env = ''
    if gcp_creds_env_key not in daemon_env:
        daemon_env += '{}={}\n'.format(gcp_creds_env_key, creds_path)
        daemon_env_path.parent.mkdir(parents=True, exist_ok=True)
        daemon_env_path.write_text(daemon_env)


def _write_openstack_snap_config(component):
    # openstack requires additional credentials setup
    openstack = endpoint_from_flag('endpoint.openstack.ready')

    cloud_config_path = _cloud_config_path(component)
    cloud_config_path.write_text('\n'.join([
        '[Global]',
        'auth-url = {}'.format(openstack.auth_url),
        'username = {}'.format(openstack.username),
        'password = {}'.format(openstack.password),
        'tenant-name = {}'.format(openstack.project_name),
        'domain-name = {}'.format(openstack.user_domain_name),
    ]))


def _write_azure_snap_config(component):
    azure = endpoint_from_flag('endpoint.azure.ready')
    cloud_config_path = _cloud_config_path(component)
    cloud_config_path.write_text(json.dumps({
        'useInstanceMetadata': True,
        'useManagedIdentityExtension': True,
        'subscriptionId': azure.subscription_id,
        'resourceGroup': azure.resource_group,
        'location': azure.resource_group_location,
        'vnetName': azure.vnet_name,
        'vnetResourceGroup': azure.vnet_resource_group,
        'subnetName': azure.subnet_name,
        'securityGroupName': azure.security_group_name,
    }))


def get_first_mount(mount_relation):
    mount_relation_list = mount_relation.mounts()
    if mount_relation_list and len(mount_relation_list) > 0:
        # mount relation list is a list of the mount layer relations
        # for now we just use the first one that is nfs
        for mount in mount_relation_list:
            # for now we just check the first mount and use that.
            # the nfs charm only supports one for now.
            if ('mounts' in mount and
                    mount['mounts'][0]['fstype'] == 'nfs'):
                return mount['mounts'][0]
    return None


@when('nfs.available')
def nfs_state_control(mount):
    ''' Determine if we should remove the state that controls the re-render
    and execution of the nfs-relation-changed event because there
    are changes in the relationship data, and we should re-render any
    configs '''

    mount_data = get_first_mount(mount)
    if mount_data:
        nfs_relation_data = {
            'options': mount_data['options'],
            'host': mount_data['hostname'],
            'mountpoint': mount_data['mountpoint'],
            'fstype': mount_data['fstype']
        }

        # Re-execute the rendering if the data has changed.
        if data_changed('nfs-config', nfs_relation_data):
            hookenv.log('reconfiguring nfs')
            remove_state('nfs.configured')


@when('nfs.available')
@when_not('nfs.configured')
def nfs_storage(mount):
    '''NFS on kubernetes requires nfs config rendered into a deployment of
    the nfs client provisioner. That will handle the persistent volume claims
    with no persistent volume to back them.'''

    mount_data = get_first_mount(mount)
    if not mount_data:
        return

    addon_path = '/root/cdk/addons/{}'
    # Render the NFS deployment
    manifest = addon_path.format('nfs-provisioner.yaml')
    render('nfs-provisioner.yaml', manifest, mount_data)
    hookenv.log('Creating the nfs provisioner.')
    try:
        kubectl('apply', '-f', manifest)
    except CalledProcessError as e:
        hookenv.log(e)
        hookenv.log('Failed to create nfs provisioner. Will attempt again next update.')  # noqa
        return

    set_state('nfs.configured')
