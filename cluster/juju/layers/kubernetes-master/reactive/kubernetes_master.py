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

import base64
import hashlib
import os
import re
import random
import shutil
import socket
import string
import json
import ipaddress

from charms.leadership import leader_get, leader_set

from shutil import move
from tempfile import TemporaryDirectory

from pathlib import Path
from shlex import split
from subprocess import check_call
from subprocess import check_output
from subprocess import CalledProcessError
from urllib.request import Request, urlopen

from charms import layer
from charms.layer import snap
from charms.reactive import hook
from charms.reactive import remove_state
from charms.reactive import set_state
from charms.reactive import is_state
from charms.reactive import endpoint_from_flag
from charms.reactive import when, when_any, when_not, when_none
from charms.reactive.helpers import data_changed, any_file_changed
from charms.kubernetes.common import get_version
from charms.kubernetes.common import retry

from charms.layer import tls_client

from charmhelpers.core import hookenv
from charmhelpers.core import host
from charmhelpers.core import unitdata
from charmhelpers.core.host import service_stop
from charmhelpers.core.templating import render
from charmhelpers.fetch import apt_install
from charmhelpers.contrib.charmsupport import nrpe


# Override the default nagios shortname regex to allow periods, which we
# need because our bin names contain them (e.g. 'snap.foo.daemon'). The
# default regex in charmhelpers doesn't allow periods, but nagios itself does.
nrpe.Check.shortname_re = '[\.A-Za-z0-9-_]+$'

gcp_creds_env_key = 'GOOGLE_APPLICATION_CREDENTIALS'
snap_resources = ['kubectl', 'kube-apiserver', 'kube-controller-manager',
                  'kube-scheduler', 'cdk-addons']

os.environ['PATH'] += os.pathsep + os.path.join(os.sep, 'snap', 'bin')
db = unitdata.kv()


def set_upgrade_needed(forced=False):
    set_state('kubernetes-master.upgrade-needed')
    config = hookenv.config()
    previous_channel = config.previous('channel')
    require_manual = config.get('require-manual-upgrade')
    hookenv.log('set upgrade needed')
    if previous_channel is None or not require_manual or forced:
        hookenv.log('forcing upgrade')
        set_state('kubernetes-master.upgrade-specified')


@when('config.changed.channel')
def channel_changed():
    set_upgrade_needed()


def service_cidr():
    ''' Return the charm's service-cidr config '''
    frozen_cidr = db.get('kubernetes-master.service-cidr')
    return frozen_cidr or hookenv.config('service-cidr')


def freeze_service_cidr():
    ''' Freeze the service CIDR. Once the apiserver has started, we can no
    longer safely change this value. '''
    db.set('kubernetes-master.service-cidr', service_cidr())


@hook('upgrade-charm')
def check_for_upgrade_needed():
    '''An upgrade charm event was triggered by Juju, react to that here.'''
    hookenv.status_set('maintenance', 'Checking resources')

    # migrate to new flags
    if is_state('kubernetes-master.restarted-for-cloud'):
        remove_state('kubernetes-master.restarted-for-cloud')
        set_state('kubernetes-master.cloud.ready')
    if is_state('kubernetes-master.cloud-request-sent'):
        # minor change, just for consistency
        remove_state('kubernetes-master.cloud-request-sent')
        set_state('kubernetes-master.cloud.request-sent')

    migrate_from_pre_snaps()
    add_rbac_roles()
    set_state('reconfigure.authentication.setup')
    remove_state('authentication.setup')

    if not db.get('snap.resources.fingerprint.initialised'):
        # We are here on an upgrade from non-rolling master
        # Since this upgrade might also include resource updates eg
        # juju upgrade-charm kubernetes-master --resource kube-any=my.snap
        # we take no risk and forcibly upgrade the snaps.
        # Forcibly means we do not prompt the user to call the upgrade action.
        set_upgrade_needed(forced=True)

    migrate_resource_checksums()
    check_resources_for_upgrade_needed()

    # Set the auto storage backend to etcd2.
    auto_storage_backend = leader_get('auto_storage_backend')
    is_leader = is_state('leadership.is_leader')
    if not auto_storage_backend and is_leader:
        leader_set(auto_storage_backend='etcd2')


def get_resource_checksum_db_key(resource):
    ''' Convert a resource name to a resource checksum database key. '''
    return 'kubernetes-master.resource-checksums.' + resource


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


def add_rbac_roles():
    '''Update the known_tokens file with proper groups.'''

    tokens_fname = '/root/cdk/known_tokens.csv'
    tokens_backup_fname = '/root/cdk/known_tokens.csv.backup'
    move(tokens_fname, tokens_backup_fname)
    with open(tokens_fname, 'w') as ftokens:
        with open(tokens_backup_fname, 'r') as stream:
            for line in stream:
                record = line.strip().split(',')
                # token, username, user, groups
                if record[2] == 'admin' and len(record) == 3:
                    towrite = '{0},{1},{2},"{3}"\n'.format(record[0],
                                                           record[1],
                                                           record[2],
                                                           'system:masters')
                    ftokens.write(towrite)
                    continue
                if record[2] == 'kube_proxy':
                    towrite = '{0},{1},{2}\n'.format(record[0],
                                                     'system:kube-proxy',
                                                     'kube-proxy')
                    ftokens.write(towrite)
                    continue
                if record[2] == 'kubelet' and record[1] == 'kubelet':
                    continue

                ftokens.write('{}'.format(line))


def rename_file_idempotent(source, destination):
    if os.path.isfile(source):
        os.rename(source, destination)


def migrate_from_pre_snaps():
    # remove old states
    remove_state('kubernetes.components.installed')
    remove_state('kubernetes.dashboard.available')
    remove_state('kube-dns.available')
    remove_state('kubernetes-master.app_version.set')

    # disable old services
    services = ['kube-apiserver',
                'kube-controller-manager',
                'kube-scheduler']
    for service in services:
        hookenv.log('Stopping {0} service.'.format(service))
        host.service_stop(service)

    # rename auth files
    os.makedirs('/root/cdk', exist_ok=True)
    rename_file_idempotent('/etc/kubernetes/serviceaccount.key',
                           '/root/cdk/serviceaccount.key')
    rename_file_idempotent('/srv/kubernetes/basic_auth.csv',
                           '/root/cdk/basic_auth.csv')
    rename_file_idempotent('/srv/kubernetes/known_tokens.csv',
                           '/root/cdk/known_tokens.csv')

    # cleanup old files
    files = [
        "/lib/systemd/system/kube-apiserver.service",
        "/lib/systemd/system/kube-controller-manager.service",
        "/lib/systemd/system/kube-scheduler.service",
        "/etc/default/kube-defaults",
        "/etc/default/kube-apiserver.defaults",
        "/etc/default/kube-controller-manager.defaults",
        "/etc/default/kube-scheduler.defaults",
        "/srv/kubernetes",
        "/home/ubuntu/kubectl",
        "/usr/local/bin/kubectl",
        "/usr/local/bin/kube-apiserver",
        "/usr/local/bin/kube-controller-manager",
        "/usr/local/bin/kube-scheduler",
        "/etc/kubernetes"
    ]
    for file in files:
        if os.path.isdir(file):
            hookenv.log("Removing directory: " + file)
            shutil.rmtree(file)
        elif os.path.isfile(file):
            hookenv.log("Removing file: " + file)
            os.remove(file)


@when('kubernetes-master.upgrade-specified')
def do_upgrade():
    install_snaps()
    remove_state('kubernetes-master.upgrade-needed')
    remove_state('kubernetes-master.upgrade-specified')


def install_snaps():
    channel = hookenv.config('channel')
    hookenv.status_set('maintenance', 'Installing kubectl snap')
    snap.install('kubectl', channel=channel, classic=True)
    hookenv.status_set('maintenance', 'Installing kube-apiserver snap')
    snap.install('kube-apiserver', channel=channel)
    hookenv.status_set('maintenance',
                       'Installing kube-controller-manager snap')
    snap.install('kube-controller-manager', channel=channel)
    hookenv.status_set('maintenance', 'Installing kube-scheduler snap')
    snap.install('kube-scheduler', channel=channel)
    hookenv.status_set('maintenance', 'Installing cdk-addons snap')
    snap.install('cdk-addons', channel=channel)
    calculate_and_store_resource_checksums()
    db.set('snap.resources.fingerprint.initialised', True)
    set_state('kubernetes-master.snaps.installed')
    remove_state('kubernetes-master.components.started')


@when('config.changed.client_password', 'leadership.is_leader')
def password_changed():
    """Handle password change via the charms config."""
    password = hookenv.config('client_password')
    if password == "" and is_state('client.password.initialised'):
        # password_changed is called during an upgrade. Nothing to do.
        return
    elif password == "":
        # Password not initialised
        password = token_generator()
    setup_basic_auth(password, "admin", "admin", "system:masters")
    set_state('reconfigure.authentication.setup')
    remove_state('authentication.setup')
    set_state('client.password.initialised')


@when('config.changed.storage-backend')
def storage_backend_changed():
    remove_state('kubernetes-master.components.started')


@when('cni.connected')
@when_not('cni.configured')
def configure_cni(cni):
    ''' Set master configuration on the CNI relation. This lets the CNI
    subordinate know that we're the master so it can respond accordingly. '''
    cni.set_config(is_master=True, kubeconfig_path='')


@when('leadership.is_leader')
@when_not('authentication.setup')
def setup_leader_authentication():
    '''Setup basic authentication and token access for the cluster.'''
    service_key = '/root/cdk/serviceaccount.key'
    basic_auth = '/root/cdk/basic_auth.csv'
    known_tokens = '/root/cdk/known_tokens.csv'

    hookenv.status_set('maintenance', 'Rendering authentication templates.')

    keys = [service_key, basic_auth, known_tokens]
    # Try first to fetch data from an old leadership broadcast.
    if not get_keys_from_leader(keys) \
            or is_state('reconfigure.authentication.setup'):
        last_pass = get_password('basic_auth.csv', 'admin')
        setup_basic_auth(last_pass, 'admin', 'admin', 'system:masters')

        if not os.path.isfile(known_tokens):
            touch(known_tokens)

        # Generate the default service account token key
        os.makedirs('/root/cdk', exist_ok=True)
        if not os.path.isfile(service_key):
            cmd = ['openssl', 'genrsa', '-out', service_key,
                   '2048']
            check_call(cmd)
        remove_state('reconfigure.authentication.setup')

    # read service account key for syndication
    leader_data = {}
    for f in [known_tokens, basic_auth, service_key]:
        with open(f, 'r') as fp:
            leader_data[f] = fp.read()

    # this is slightly opaque, but we are sending file contents under its file
    # path as a key.
    # eg:
    # {'/root/cdk/serviceaccount.key': 'RSA:2471731...'}
    leader_set(leader_data)
    remove_state('kubernetes-master.components.started')
    set_state('authentication.setup')


@when_not('leadership.is_leader')
def setup_non_leader_authentication():

    service_key = '/root/cdk/serviceaccount.key'
    basic_auth = '/root/cdk/basic_auth.csv'
    known_tokens = '/root/cdk/known_tokens.csv'

    keys = [service_key, basic_auth, known_tokens]
    # The source of truth for non-leaders is the leader.
    # Therefore we overwrite_local with whatever the leader has.
    if not get_keys_from_leader(keys, overwrite_local=True):
        # the keys were not retrieved. Non-leaders have to retry.
        return

    if not any_file_changed(keys) and is_state('authentication.setup'):
        # No change detected and we have already setup the authentication
        return

    hookenv.status_set('maintenance', 'Rendering authentication templates.')

    remove_state('kubernetes-master.components.started')
    set_state('authentication.setup')


def get_keys_from_leader(keys, overwrite_local=False):
    """
    Gets the broadcasted keys from the leader and stores them in
    the corresponding files.

    Args:
        keys: list of keys. Keys are actually files on the FS.

    Returns: True if all key were fetched, False if not.

    """
    # This races with other codepaths, and seems to require being created first
    # This block may be extracted later, but for now seems to work as intended
    os.makedirs('/root/cdk', exist_ok=True)

    for k in keys:
        # If the path does not exist, assume we need it
        if not os.path.exists(k) or overwrite_local:
            # Fetch data from leadership broadcast
            contents = leader_get(k)
            # Default to logging the warning and wait for leader data to be set
            if contents is None:
                hookenv.log('Missing content for file {}'.format(k))
                return False
            # Write out the file and move on to the next item
            with open(k, 'w+') as fp:
                fp.write(contents)
                fp.write('\n')

    return True


@when('kubernetes-master.snaps.installed')
def set_app_version():
    ''' Declare the application version to juju '''
    version = check_output(['kube-apiserver', '--version'])
    hookenv.application_version_set(version.split(b' v')[-1].rstrip())


@when('kubernetes-master.snaps.installed')
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
    if data_changed('master_snapd_refresh', timer):
        hookenv.log('setting snapd_refresh timer to: {}'.format(timer))
        leader_set({'snapd_refresh': timer})


@when('kubernetes-master.snaps.installed')
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
def set_final_status():
    ''' Set the final status of the charm as we leave hook execution '''
    try:
        goal_state = hookenv.goal_state()
    except NotImplementedError:
        goal_state = {}

    vsphere_joined = is_state('endpoint.vsphere.joined')
    azure_joined = is_state('endpoint.azure.joined')
    cloud_blocked = is_state('kubernetes-master.cloud.blocked')
    if vsphere_joined and cloud_blocked:
        hookenv.status_set('blocked',
                           'vSphere integration requires K8s 1.12 or greater')
        return
    if azure_joined and cloud_blocked:
        hookenv.status_set('blocked',
                           'Azure integration requires K8s 1.11 or greater')
        return

    if is_state('kubernetes-master.cloud.pending'):
        hookenv.status_set('waiting', 'Waiting for cloud integration')
        return

    if not is_state('kube-api-endpoint.available'):
        if 'kube-api-endpoint' in goal_state.get('relations', {}):
            status = 'waiting'
        else:
            status = 'blocked'
        hookenv.status_set(status, 'Waiting for kube-api-endpoint relation')
        return

    if not is_state('kube-control.connected'):
        if 'kube-control' in goal_state.get('relations', {}):
            status = 'waiting'
        else:
            status = 'blocked'
        hookenv.status_set(status, 'Waiting for workers.')
        return

    upgrade_needed = is_state('kubernetes-master.upgrade-needed')
    upgrade_specified = is_state('kubernetes-master.upgrade-specified')
    if upgrade_needed and not upgrade_specified:
        msg = 'Needs manual upgrade, run the upgrade action'
        hookenv.status_set('blocked', msg)
        return

    if is_state('kubernetes-master.components.started'):
        # All services should be up and running at this point. Double-check...
        failing_services = master_services_down()
        if len(failing_services) != 0:
            msg = 'Stopped services: {}'.format(','.join(failing_services))
            hookenv.status_set('blocked', msg)
            return

    is_leader = is_state('leadership.is_leader')
    authentication_setup = is_state('authentication.setup')
    if not is_leader and not authentication_setup:
        hookenv.status_set('waiting', 'Waiting on leaders crypto keys.')
        return

    components_started = is_state('kubernetes-master.components.started')
    addons_configured = is_state('cdk-addons.configured')
    if components_started and not addons_configured:
        hookenv.status_set('waiting', 'Waiting to retry addon deployment')
        return

    if addons_configured and not all_kube_system_pods_running():
        hookenv.status_set('waiting', 'Waiting for kube-system pods to start')
        return

    if hookenv.config('service-cidr') != service_cidr():
        msg = 'WARN: cannot change service-cidr, still using ' + service_cidr()
        hookenv.status_set('active', msg)
        return

    gpu_available = is_state('kube-control.gpu.available')
    gpu_enabled = is_state('kubernetes-master.gpu.enabled')
    if gpu_available and not gpu_enabled:
        msg = 'GPUs available. Set allow-privileged="auto" to enable.'
        hookenv.status_set('active', msg)
        return

    hookenv.status_set('active', 'Kubernetes master running.')


def master_services_down():
    """Ensure master services are up and running.

    Return: list of failing services"""
    services = ['kube-apiserver',
                'kube-controller-manager',
                'kube-scheduler']
    failing_services = []
    for service in services:
        daemon = 'snap.{}.daemon'.format(service)
        if not host.service_running(daemon):
            failing_services.append(service)
    return failing_services


@when('etcd.available', 'tls_client.server.certificate.saved',
      'authentication.setup')
@when('leadership.set.auto_storage_backend')
@when_not('kubernetes-master.components.started',
          'kubernetes-master.cloud.pending',
          'kubernetes-master.cloud.blocked')
def start_master(etcd):
    '''Run the Kubernetes master components.'''
    hookenv.status_set('maintenance',
                       'Configuring the Kubernetes master services.')
    freeze_service_cidr()
    if not etcd.get_connection_string():
        # etcd is not returning a connection string. This happens when
        # the master unit disconnects from etcd and is ready to terminate.
        # No point in trying to start master services and fail. Just return.
        return

    # TODO: Make sure below relation is handled on change
    # https://github.com/kubernetes/kubernetes/issues/43461
    handle_etcd_relation(etcd)

    # Add CLI options to all components
    configure_apiserver(etcd.get_connection_string())
    configure_controller_manager()
    configure_scheduler()
    set_state('kubernetes-master.components.started')
    hookenv.open_port(6443)


@when('etcd.available')
def etcd_data_change(etcd):
    ''' Etcd scale events block master reconfiguration due to the
        kubernetes-master.components.started state. We need a way to
        handle these events consistently only when the number of etcd
        units has actually changed '''

    # key off of the connection string
    connection_string = etcd.get_connection_string()

    # If the connection string changes, remove the started state to trigger
    # handling of the master components
    if data_changed('etcd-connect', connection_string):
        remove_state('kubernetes-master.components.started')

    # We are the leader and the auto_storage_backend is not set meaning
    # this is the first time we connect to etcd.
    auto_storage_backend = leader_get('auto_storage_backend')
    is_leader = is_state('leadership.is_leader')
    if is_leader and not auto_storage_backend:
        if etcd.get_version().startswith('3.'):
            leader_set(auto_storage_backend='etcd3')
        else:
            leader_set(auto_storage_backend='etcd2')


@when('kube-control.connected')
@when('cdk-addons.configured')
def send_cluster_dns_detail(kube_control):
    ''' Send cluster DNS info '''
    enableKubeDNS = hookenv.config('enable-kube-dns')
    dnsDomain = hookenv.config('dns_domain')
    dns_ip = None
    if enableKubeDNS:
        try:
            dns_ip = get_dns_ip()
        except CalledProcessError:
            hookenv.log("kubedns not ready yet")
            return
    kube_control.set_dns(53, dnsDomain, dns_ip, enableKubeDNS)


@when('kube-control.connected')
@when('snap.installed.kubectl')
@when('leadership.is_leader')
def create_service_configs(kube_control):
    """Create the users for kubelet"""
    should_restart = False
    # generate the username/pass for the requesting unit
    proxy_token = get_token('system:kube-proxy')
    if not proxy_token:
        setup_tokens(None, 'system:kube-proxy', 'kube-proxy')
        proxy_token = get_token('system:kube-proxy')
        should_restart = True

    client_token = get_token('admin')
    if not client_token:
        setup_tokens(None, 'admin', 'admin', "system:masters")
        client_token = get_token('admin')
        should_restart = True

    requests = kube_control.auth_user()
    for request in requests:
        username = request[1]['user']
        group = request[1]['group']
        kubelet_token = get_token(username)
        if not kubelet_token and username and group:
            # Usernames have to be in the form of system:node:<nodeName>
            userid = "kubelet-{}".format(request[0].split('/')[1])
            setup_tokens(None, username, userid, group)
            kubelet_token = get_token(username)
            kube_control.sign_auth_request(request[0], username,
                                           kubelet_token, proxy_token,
                                           client_token)
            should_restart = True

    if should_restart:
        host.service_restart('snap.kube-apiserver.daemon')
        remove_state('authentication.setup')


@when('kube-api-endpoint.available')
def push_service_data(kube_api):
    ''' Send configuration to the load balancer, and close access to the
    public interface '''
    kube_api.configure(port=6443)


def get_ingress_address(relation_name):
    try:
        network_info = hookenv.network_get(relation_name)
    except NotImplementedError:
        network_info = []

    if network_info and 'ingress-addresses' in network_info:
        # just grab the first one for now, maybe be more robust here?
        return network_info['ingress-addresses'][0]
    else:
        # if they don't have ingress-addresses they are running a juju that
        # doesn't support spaces, so just return the private address
        return hookenv.unit_get('private-address')


@when('certificates.available', 'kube-api-endpoint.available')
def send_data(tls, kube_api_endpoint):
    '''Send the data that is required to create a server certificate for
    this server.'''
    # Use the public ip of this unit as the Common Name for the certificate.
    common_name = hookenv.unit_public_ip()

    # Get the SDN gateway based on the cidr address.
    kubernetes_service_ip = get_kubernetes_service_ip()

    # Get ingress address
    ingress_ip = get_ingress_address(kube_api_endpoint.relation_name)

    domain = hookenv.config('dns_domain')
    # Create SANs that the tls layer will add to the server cert.
    sans = [
        hookenv.unit_public_ip(),
        ingress_ip,
        socket.gethostname(),
        kubernetes_service_ip,
        'kubernetes',
        'kubernetes.{0}'.format(domain),
        'kubernetes.default',
        'kubernetes.default.svc',
        'kubernetes.default.svc.{0}'.format(domain)
    ]

    # maybe they have extra names they want as SANs
    extra_sans = hookenv.config('extra_sans')
    if extra_sans and not extra_sans == "":
        sans.extend(extra_sans.split())

    # Create a path safe name by removing path characters from the unit name.
    certificate_name = hookenv.local_unit().replace('/', '_')
    # Request a server cert with this information.
    tls.request_server_cert(common_name, sans, certificate_name)


@when('config.changed.extra_sans', 'certificates.available',
      'kube-api-endpoint.available')
def update_certificate(tls, kube_api_endpoint):
    # Using the config.changed.extra_sans flag to catch changes.
    # IP changes will take ~5 minutes or so to propagate, but
    # it will update.
    send_data(tls, kube_api_endpoint)


@when('certificates.server.cert.available',
      'kubernetes-master.components.started',
      'tls_client.server.certificate.written')
def kick_api_server(tls):
    # need to be idempotent and don't want to kick the api server
    # without need
    if data_changed('cert', tls.get_server_cert()):
        # certificate changed, so restart the api server
        hookenv.log("Certificate information changed, restarting api server")
        restart_apiserver()
    tls_client.reset_certificate_write_flag('server')


@when_any('kubernetes-master.components.started', 'ceph-storage.configured')
@when('leadership.is_leader')
def configure_cdk_addons():
    ''' Configure CDK addons '''
    remove_state('cdk-addons.configured')
    load_gpu_plugin = hookenv.config('enable-nvidia-plugin').lower()
    gpuEnable = (get_version('kube-apiserver') >= (1, 9) and
                 load_gpu_plugin == "auto" and
                 is_state('kubernetes-master.gpu.enabled'))
    registry = hookenv.config('addons-registry')
    dbEnabled = str(hookenv.config('enable-dashboard-addons')).lower()
    dnsEnabled = str(hookenv.config('enable-kube-dns')).lower()
    metricsEnabled = str(hookenv.config('enable-metrics')).lower()
    if (is_state('ceph-storage.configured') and
            get_version('kube-apiserver') >= (1, 10)):
        cephEnabled = "true"
    else:
        cephEnabled = "false"
    ceph_ep = endpoint_from_flag('ceph-storage.available')
    ceph = {}
    default_storage = ''
    if ceph_ep:
        b64_ceph_key = base64.b64encode(ceph_ep.key().encode('utf-8'))
        ceph['admin_key'] = b64_ceph_key.decode('ascii')
        ceph['kubernetes_key'] = b64_ceph_key.decode('ascii')
        ceph['mon_hosts'] = ceph_ep.mon_hosts()
        default_storage = hookenv.config('default-storage')

    args = [
        'arch=' + arch(),
        'dns-ip=' + get_deprecated_dns_ip(),
        'dns-domain=' + hookenv.config('dns_domain'),
        'registry=' + registry,
        'enable-dashboard=' + dbEnabled,
        'enable-kube-dns=' + dnsEnabled,
        'enable-metrics=' + metricsEnabled,
        'enable-gpu=' + str(gpuEnable).lower(),
        'enable-ceph=' + cephEnabled,
        'ceph-admin-key=' + (ceph.get('admin_key', '')),
        'ceph-kubernetes-key=' + (ceph.get('admin_key', '')),
        'ceph-mon-hosts="' + (ceph.get('mon_hosts', '')) + '"',
        'default-storage=' + default_storage,
    ]
    check_call(['snap', 'set', 'cdk-addons'] + args)
    if not addons_ready():
        remove_state('cdk-addons.configured')
        return

    set_state('cdk-addons.configured')


@retry(times=3, delay_secs=20)
def addons_ready():
    """
    Test if the add ons got installed

    Returns: True is the addons got applied

    """
    try:
        check_call(['cdk-addons.apply'])
        return True
    except CalledProcessError:
        hookenv.log("Addons are not ready yet.")
        return False


@when('loadbalancer.available', 'certificates.ca.available',
      'certificates.client.cert.available', 'authentication.setup')
def loadbalancer_kubeconfig(loadbalancer, ca, client):
    # Get the potential list of loadbalancers from the relation object.
    hosts = loadbalancer.get_addresses_ports()
    # Get the public address of loadbalancers so users can access the cluster.
    address = hosts[0].get('public-address')
    # Get the port of the loadbalancer so users can access the cluster.
    port = hosts[0].get('port')
    server = 'https://{0}:{1}'.format(address, port)
    build_kubeconfig(server)


@when('certificates.ca.available', 'certificates.client.cert.available',
      'authentication.setup')
@when_not('loadbalancer.available')
def create_self_config(ca, client):
    '''Create a kubernetes configuration for the master unit.'''
    server = 'https://{0}:{1}'.format(hookenv.unit_get('public-address'), 6443)
    build_kubeconfig(server)


@when('ceph-storage.available')
def ceph_state_control(ceph_admin):
    ''' Determine if we should remove the state that controls the re-render
    and execution of the ceph-relation-changed event because there
    are changes in the relationship data, and we should re-render any
    configs, keys, and/or service pre-reqs '''

    ceph_relation_data = {
        'mon_hosts': ceph_admin.mon_hosts(),
        'fsid': ceph_admin.fsid(),
        'auth_supported': ceph_admin.auth(),
        'hostname': socket.gethostname(),
        'key': ceph_admin.key()
    }

    # Re-execute the rendering if the data has changed.
    if data_changed('ceph-config', ceph_relation_data):
        remove_state('ceph-storage.configured')


@when('ceph-storage.available')
@when_not('ceph-storage.configured')
def ceph_storage(ceph_admin):
    '''Ceph on kubernetes will require a few things - namely a ceph
    configuration, and the ceph secret key file used for authentication.
    This method will install the client package, and render the requisit files
    in order to consume the ceph-storage relation.'''

    # deprecated in 1.10 in favor of using CSI
    if get_version('kube-apiserver') >= (1, 10):
        # this is actually false, but by setting this flag we won't keep
        # running this function for no reason. Also note that we watch this
        # flag to run cdk-addons.apply.
        set_state('ceph-storage.configured')
        return

    ceph_context = {
        'mon_hosts': ceph_admin.mon_hosts(),
        'fsid': ceph_admin.fsid(),
        'auth_supported': ceph_admin.auth(),
        'use_syslog': "true",
        'ceph_public_network': '',
        'ceph_cluster_network': '',
        'loglevel': 1,
        'hostname': socket.gethostname(),
    }
    # Install the ceph common utilities.
    apt_install(['ceph-common'], fatal=True)

    etc_ceph_directory = '/etc/ceph'
    if not os.path.isdir(etc_ceph_directory):
        os.makedirs(etc_ceph_directory)
    charm_ceph_conf = os.path.join(etc_ceph_directory, 'ceph.conf')
    # Render the ceph configuration from the ceph conf template
    render('ceph.conf', charm_ceph_conf, ceph_context)

    # The key can rotate independently of other ceph config, so validate it
    admin_key = os.path.join(etc_ceph_directory,
                             'ceph.client.admin.keyring')
    try:
        with open(admin_key, 'w') as key_file:
            key_file.write("[client.admin]\n\tkey = {}\n".format(
                ceph_admin.key()))
    except IOError as err:
        hookenv.log("IOError writing admin.keyring: {}".format(err))

    # Enlist the ceph-admin key as a kubernetes secret
    if ceph_admin.key():
        encoded_key = base64.b64encode(ceph_admin.key().encode('utf-8'))
    else:
        # We didn't have a key, and cannot proceed. Do not set state and
        # allow this method to re-execute
        return

    context = {'secret': encoded_key.decode('ascii')}
    render('ceph-secret.yaml', '/tmp/ceph-secret.yaml', context)
    try:
        # At first glance this is deceptive. The apply stanza will create if
        # it doesn't exist, otherwise it will update the entry, ensuring our
        # ceph-secret is always reflective of what we have in /etc/ceph
        # assuming we have invoked this anytime that file would change.
        cmd = ['kubectl', 'apply', '-f', '/tmp/ceph-secret.yaml']
        check_call(cmd)
        os.remove('/tmp/ceph-secret.yaml')
    except:  # NOQA
        # the enlistment in kubernetes failed, return and prepare for re-exec
        return

    # when complete, set a state relating to configuration of the storage
    # backend that will allow other modules to hook into this and verify we
    # have performed the necessary pre-req steps to interface with a ceph
    # deployment.
    set_state('ceph-storage.configured')


@when('nrpe-external-master.available')
@when_not('nrpe-external-master.initial-config')
def initial_nrpe_config(nagios=None):
    set_state('nrpe-external-master.initial-config')
    update_nrpe_config(nagios)


@when('config.changed.authorization-mode',
      'kubernetes-master.components.started')
def switch_auth_mode():
    config = hookenv.config()
    mode = config.get('authorization-mode')
    if data_changed('auth-mode', mode):
        remove_state('kubernetes-master.components.started')


@when('kubernetes-master.components.started')
@when('nrpe-external-master.available')
@when_any('config.changed.nagios_context',
          'config.changed.nagios_servicegroups')
def update_nrpe_config(unused=None):
    services = (
        'snap.kube-apiserver.daemon',
        'snap.kube-controller-manager.daemon',
        'snap.kube-scheduler.daemon'
    )
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
    services = (
        'snap.kube-apiserver.daemon',
        'snap.kube-controller-manager.daemon',
        'snap.kube-scheduler.daemon'
    )

    # The current nrpe-external-master interface doesn't handle a lot of logic,
    # use the charm-helpers code for now.
    hostname = nrpe.get_nagios_hostname()
    nrpe_setup = nrpe.NRPE(hostname=hostname)

    for service in services:
        nrpe_setup.remove_check(shortname=service)


def is_privileged():
    """Return boolean indicating whether or not to set allow-privileged=true.

    """
    privileged = hookenv.config('allow-privileged').lower()
    if privileged == 'auto':
        return is_state('kubernetes-master.gpu.enabled')
    else:
        return privileged == 'true'


@when('config.changed.allow-privileged')
@when('kubernetes-master.components.started')
def on_config_allow_privileged_change():
    """React to changed 'allow-privileged' config value.

    """
    remove_state('kubernetes-master.components.started')
    remove_state('config.changed.allow-privileged')


@when_any('config.changed.api-extra-args',
          'config.changed.audit-policy',
          'config.changed.audit-webhook-config')
@when('kubernetes-master.components.started')
@when('leadership.set.auto_storage_backend')
@when('etcd.available')
def reconfigure_apiserver(etcd):
    configure_apiserver(etcd.get_connection_string())


@when('config.changed.controller-manager-extra-args')
@when('kubernetes-master.components.started')
def on_config_controller_manager_extra_args_change():
    configure_controller_manager()


@when('config.changed.scheduler-extra-args')
@when('kubernetes-master.components.started')
def on_config_scheduler_extra_args_change():
    configure_scheduler()


@when('kube-control.gpu.available')
@when('kubernetes-master.components.started')
@when_not('kubernetes-master.gpu.enabled')
def on_gpu_available(kube_control):
    """The remote side (kubernetes-worker) is gpu-enabled.

    We need to run in privileged mode.

    """
    kube_version = get_version('kube-apiserver')
    config = hookenv.config()
    if (config['allow-privileged'].lower() == "false" and
            kube_version < (1, 9)):
        return

    remove_state('kubernetes-master.components.started')
    set_state('kubernetes-master.gpu.enabled')


@when('kubernetes-master.gpu.enabled')
@when('kubernetes-master.components.started')
@when_not('kubernetes-master.privileged')
def gpu_with_no_privileged():
    """We were in gpu mode, but the operator has set allow-privileged="false",
    so we can't run in gpu mode anymore.

    """
    if get_version('kube-apiserver') < (1, 9):
        remove_state('kubernetes-master.gpu.enabled')


@when('kube-control.connected')
@when_not('kube-control.gpu.available')
@when('kubernetes-master.gpu.enabled')
@when('kubernetes-master.components.started')
def gpu_departed(kube_control):
    """We were in gpu mode, but the workers informed us there is
    no gpu support anymore.

    """
    remove_state('kubernetes-master.gpu.enabled')


@hook('stop')
def shutdown():
    """ Stop the kubernetes master services

    """
    service_stop('snap.kube-apiserver.daemon')
    service_stop('snap.kube-controller-manager.daemon')
    service_stop('snap.kube-scheduler.daemon')


def restart_apiserver():
    hookenv.status_set('maintenance', 'Restarting kube-apiserver')
    host.service_restart('snap.kube-apiserver.daemon')


def restart_controller_manager():
    hookenv.status_set('maintenance', 'Restarting kube-controller-manager')
    host.service_restart('snap.kube-controller-manager.daemon')


def restart_scheduler():
    hookenv.status_set('maintenance', 'Restarting kube-scheduler')
    host.service_restart('snap.kube-scheduler.daemon')


def arch():
    '''Return the package architecture as a string. Raise an exception if the
    architecture is not supported by kubernetes.'''
    # Get the package architecture for this system.
    architecture = check_output(['dpkg', '--print-architecture']).rstrip()
    # Convert the binary result into a string.
    architecture = architecture.decode('utf-8')
    return architecture


def build_kubeconfig(server):
    '''Gather the relevant data for Kubernetes configuration objects and create
    a config object with that information.'''
    # Get the options from the tls-client layer.
    layer_options = layer.options('tls-client')
    # Get all the paths to the tls information required for kubeconfig.
    ca = layer_options.get('ca_certificate_path')
    ca_exists = ca and os.path.isfile(ca)
    client_pass = get_password('basic_auth.csv', 'admin')
    # Do we have everything we need?
    if ca_exists and client_pass:
        # Create an absolute path for the kubeconfig file.
        kubeconfig_path = os.path.join(os.sep, 'home', 'ubuntu', 'config')
        # Create the kubeconfig on this system so users can access the cluster.

        create_kubeconfig(kubeconfig_path, server, ca,
                          user='admin', password=client_pass)
        # Make the config file readable by the ubuntu users so juju scp works.
        cmd = ['chown', 'ubuntu:ubuntu', kubeconfig_path]
        check_call(cmd)


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


def get_dns_ip():
    cmd = "kubectl get service --namespace kube-system kube-dns --output json"
    output = check_output(cmd, shell=True).decode()
    svc = json.loads(output)
    return svc['spec']['clusterIP']


def get_deprecated_dns_ip():
    '''We previously hardcoded the dns ip. This function returns the old
    hardcoded value for use with older versions of cdk_addons.'''
    interface = ipaddress.IPv4Interface(service_cidr())
    ip = interface.network.network_address + 10
    return ip.exploded


def get_kubernetes_service_ip():
    '''Get the IP address for the kubernetes service based on the cidr.'''
    interface = ipaddress.IPv4Interface(service_cidr())
    # Add .1 at the end of the network
    ip = interface.network.network_address + 1
    return ip.exploded


def handle_etcd_relation(reldata):
    ''' Save the client credentials and set appropriate daemon flags when
    etcd declares itself as available'''
    # Define where the etcd tls files will be kept.
    etcd_dir = '/root/cdk/etcd'

    # Create paths to the etcd client ca, key, and cert file locations.
    ca = os.path.join(etcd_dir, 'client-ca.pem')
    key = os.path.join(etcd_dir, 'client-key.pem')
    cert = os.path.join(etcd_dir, 'client-cert.pem')

    # Save the client credentials (in relation data) to the paths provided.
    reldata.save_client_credentials(key, cert, ca)


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
    prev_args_key = 'kubernetes-master.prev_args.' + service
    prev_args = db.get(prev_args_key) or {}

    extra_args = parse_extra_args(extra_args_key)

    args = {}
    for arg in prev_args:
        # remove previous args by setting to null
        # note this is so we remove them from the snap's config
        args[arg] = 'null'
    for k, v in base_args.items():
        args[k] = v
    for k, v in extra_args.items():
        args[k] = v

    cmd = ['snap', 'set', service] + ['%s=%s' % item for item in args.items()]
    check_call(cmd)

    db.set(prev_args_key, args)


def remove_if_exists(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def write_audit_config_file(path, contents):
    with open(path, 'w') as f:
        header = '# Autogenerated by kubernetes-master charm'
        f.write(header + '\n' + contents)


def configure_apiserver(etcd_connection_string):
    api_opts = {}

    # Get the tls paths from the layer data.
    layer_options = layer.options('tls-client')
    ca_cert_path = layer_options.get('ca_certificate_path')
    client_cert_path = layer_options.get('client_certificate_path')
    client_key_path = layer_options.get('client_key_path')
    server_cert_path = layer_options.get('server_certificate_path')
    server_key_path = layer_options.get('server_key_path')

    # at one point in time, this code would set ca-client-cert,
    # but this was removed. This was before configure_kubernetes_service
    # kept track of old arguments and removed them, so client-ca-cert
    # was able to hang around forever stored in the snap configuration.
    # This removes that stale configuration from the snap if it still
    # exists.
    api_opts['client-ca-file'] = 'null'

    if is_privileged():
        api_opts['allow-privileged'] = 'true'
        set_state('kubernetes-master.privileged')
    else:
        api_opts['allow-privileged'] = 'false'
        remove_state('kubernetes-master.privileged')

    # Handle static options for now
    api_opts['service-cluster-ip-range'] = service_cidr()
    api_opts['min-request-timeout'] = '300'
    api_opts['v'] = '4'
    api_opts['tls-cert-file'] = server_cert_path
    api_opts['tls-private-key-file'] = server_key_path
    api_opts['kubelet-certificate-authority'] = ca_cert_path
    api_opts['kubelet-client-certificate'] = client_cert_path
    api_opts['kubelet-client-key'] = client_key_path
    api_opts['logtostderr'] = 'true'
    api_opts['insecure-bind-address'] = '127.0.0.1'
    api_opts['insecure-port'] = '8080'
    api_opts['storage-backend'] = getStorageBackend()
    api_opts['basic-auth-file'] = '/root/cdk/basic_auth.csv'

    api_opts['token-auth-file'] = '/root/cdk/known_tokens.csv'
    api_opts['service-account-key-file'] = '/root/cdk/serviceaccount.key'
    api_opts['kubelet-preferred-address-types'] = \
        '[InternalIP,Hostname,InternalDNS,ExternalDNS,ExternalIP]'
    api_opts['advertise-address'] = get_ingress_address('kube-control')

    etcd_dir = '/root/cdk/etcd'
    etcd_ca = os.path.join(etcd_dir, 'client-ca.pem')
    etcd_key = os.path.join(etcd_dir, 'client-key.pem')
    etcd_cert = os.path.join(etcd_dir, 'client-cert.pem')

    api_opts['etcd-cafile'] = etcd_ca
    api_opts['etcd-keyfile'] = etcd_key
    api_opts['etcd-certfile'] = etcd_cert
    api_opts['etcd-servers'] = etcd_connection_string

    admission_control_pre_1_9 = [
        'NamespaceLifecycle',
        'LimitRanger',
        'ServiceAccount',
        'ResourceQuota',
        'DefaultTolerationSeconds'
    ]

    admission_control = [
        'NamespaceLifecycle',
        'LimitRanger',
        'ServiceAccount',
        'PersistentVolumeLabel',
        'DefaultStorageClass',
        'DefaultTolerationSeconds',
        'MutatingAdmissionWebhook',
        'ValidatingAdmissionWebhook',
        'ResourceQuota'
    ]

    auth_mode = hookenv.config('authorization-mode')
    if 'Node' in auth_mode:
        admission_control.append('NodeRestriction')

    api_opts['authorization-mode'] = auth_mode

    kube_version = get_version('kube-apiserver')
    if kube_version < (1, 6):
        hookenv.log('Removing DefaultTolerationSeconds from admission-control')
        admission_control_pre_1_9.remove('DefaultTolerationSeconds')
    if kube_version < (1, 9):
        api_opts['admission-control'] = ','.join(admission_control_pre_1_9)
    else:
        api_opts['admission-control'] = ','.join(admission_control)

    if kube_version > (1, 6) and \
       hookenv.config('enable-metrics'):
        api_opts['requestheader-client-ca-file'] = ca_cert_path
        api_opts['requestheader-allowed-names'] = 'client'
        api_opts['requestheader-extra-headers-prefix'] = 'X-Remote-Extra-'
        api_opts['requestheader-group-headers'] = 'X-Remote-Group'
        api_opts['requestheader-username-headers'] = 'X-Remote-User'
        api_opts['proxy-client-cert-file'] = client_cert_path
        api_opts['proxy-client-key-file'] = client_key_path
        api_opts['enable-aggregator-routing'] = 'true'
        api_opts['client-ca-file'] = ca_cert_path

    if is_state('endpoint.aws.ready'):
        api_opts['cloud-provider'] = 'aws'
    elif is_state('endpoint.gcp.ready'):
        cloud_config_path = _cloud_config_path('kube-apiserver')
        api_opts['cloud-provider'] = 'gce'
        api_opts['cloud-config'] = str(cloud_config_path)
    elif is_state('endpoint.openstack.ready'):
        cloud_config_path = _cloud_config_path('kube-apiserver')
        api_opts['cloud-provider'] = 'openstack'
        api_opts['cloud-config'] = str(cloud_config_path)
    elif (is_state('endpoint.vsphere.ready') and
          get_version('kube-apiserver') >= (1, 12)):
        cloud_config_path = _cloud_config_path('kube-apiserver')
        api_opts['cloud-provider'] = 'vsphere'
        api_opts['cloud-config'] = str(cloud_config_path)
    elif is_state('endpoint.azure.ready'):
        cloud_config_path = _cloud_config_path('kube-apiserver')
        api_opts['cloud-provider'] = 'azure'
        api_opts['cloud-config'] = str(cloud_config_path)

    audit_root = '/root/cdk/audit'
    os.makedirs(audit_root, exist_ok=True)

    audit_log_path = audit_root + '/audit.log'
    api_opts['audit-log-path'] = audit_log_path
    api_opts['audit-log-maxsize'] = '100'
    api_opts['audit-log-maxbackup'] = '9'

    audit_policy_path = audit_root + '/audit-policy.yaml'
    audit_policy = hookenv.config('audit-policy')
    if audit_policy:
        write_audit_config_file(audit_policy_path, audit_policy)
        api_opts['audit-policy-file'] = audit_policy_path
    else:
        remove_if_exists(audit_policy_path)

    audit_webhook_config_path = audit_root + '/audit-webhook-config.yaml'
    audit_webhook_config = hookenv.config('audit-webhook-config')
    if audit_webhook_config:
        write_audit_config_file(audit_webhook_config_path,
                                audit_webhook_config)
        api_opts['audit-webhook-config-file'] = audit_webhook_config_path
    else:
        remove_if_exists(audit_webhook_config_path)

    configure_kubernetes_service('kube-apiserver', api_opts, 'api-extra-args')
    restart_apiserver()


def configure_controller_manager():
    controller_opts = {}

    # Get the tls paths from the layer data.
    layer_options = layer.options('tls-client')
    ca_cert_path = layer_options.get('ca_certificate_path')

    # Default to 3 minute resync. TODO: Make this configurable?
    controller_opts['min-resync-period'] = '3m'
    controller_opts['v'] = '2'
    controller_opts['root-ca-file'] = ca_cert_path
    controller_opts['logtostderr'] = 'true'
    controller_opts['master'] = 'http://127.0.0.1:8080'

    controller_opts['service-account-private-key-file'] = \
        '/root/cdk/serviceaccount.key'

    if is_state('endpoint.aws.ready'):
        controller_opts['cloud-provider'] = 'aws'
    elif is_state('endpoint.gcp.ready'):
        cloud_config_path = _cloud_config_path('kube-controller-manager')
        controller_opts['cloud-provider'] = 'gce'
        controller_opts['cloud-config'] = str(cloud_config_path)
    elif is_state('endpoint.openstack.ready'):
        cloud_config_path = _cloud_config_path('kube-controller-manager')
        controller_opts['cloud-provider'] = 'openstack'
        controller_opts['cloud-config'] = str(cloud_config_path)
    elif (is_state('endpoint.vsphere.ready') and
          get_version('kube-apiserver') >= (1, 12)):
        cloud_config_path = _cloud_config_path('kube-controller-manager')
        controller_opts['cloud-provider'] = 'vsphere'
        controller_opts['cloud-config'] = str(cloud_config_path)
    elif is_state('endpoint.azure.ready'):
        cloud_config_path = _cloud_config_path('kube-controller-manager')
        controller_opts['cloud-provider'] = 'azure'
        controller_opts['cloud-config'] = str(cloud_config_path)

    configure_kubernetes_service('kube-controller-manager', controller_opts,
                                 'controller-manager-extra-args')
    restart_controller_manager()


def configure_scheduler():
    scheduler_opts = {}

    scheduler_opts['v'] = '2'
    scheduler_opts['logtostderr'] = 'true'
    scheduler_opts['master'] = 'http://127.0.0.1:8080'

    configure_kubernetes_service('kube-scheduler', scheduler_opts,
                                 'scheduler-extra-args')

    restart_scheduler()


def setup_basic_auth(password=None, username='admin', uid='admin',
                     groups=None):
    '''Create the htacces file and the tokens.'''
    root_cdk = '/root/cdk'
    if not os.path.isdir(root_cdk):
        os.makedirs(root_cdk)
    htaccess = os.path.join(root_cdk, 'basic_auth.csv')
    if not password:
        password = token_generator()
    with open(htaccess, 'w') as stream:
        if groups:
            stream.write('{0},{1},{2},"{3}"'.format(password,
                                                    username, uid, groups))
        else:
            stream.write('{0},{1},{2}'.format(password, username, uid))


def setup_tokens(token, username, user, groups=None):
    '''Create a token file for kubernetes authentication.'''
    root_cdk = '/root/cdk'
    if not os.path.isdir(root_cdk):
        os.makedirs(root_cdk)
    known_tokens = os.path.join(root_cdk, 'known_tokens.csv')
    if not token:
        token = token_generator()
    with open(known_tokens, 'a') as stream:
        if groups:
            stream.write('{0},{1},{2},"{3}"\n'.format(token,
                                                      username,
                                                      user,
                                                      groups))
        else:
            stream.write('{0},{1},{2}\n'.format(token, username, user))


def get_password(csv_fname, user):
    '''Get the password of user within the csv file provided.'''
    root_cdk = '/root/cdk'
    tokens_fname = os.path.join(root_cdk, csv_fname)
    if not os.path.isfile(tokens_fname):
        return None
    with open(tokens_fname, 'r') as stream:
        for line in stream:
            record = line.split(',')
            if record[1] == user:
                return record[0]
    return None


def get_token(username):
    """Grab a token from the static file if present. """
    return get_password('known_tokens.csv', username)


def set_token(password, save_salt):
    ''' Store a token so it can be recalled later by token_generator.

    param: password - the password to be stored
    param: save_salt - the key to store the value of the token.'''
    db.set(save_salt, password)
    return db.get(save_salt)


def token_generator(length=32):
    ''' Generate a random token for use in passwords and account tokens.

    param: length - the length of the token to generate'''
    alpha = string.ascii_letters + string.digits
    token = ''.join(random.SystemRandom().choice(alpha) for _ in range(length))
    return token


@retry(times=3, delay_secs=10)
def all_kube_system_pods_running():
    ''' Check pod status in the kube-system namespace. Returns True if all
    pods are running, False otherwise. '''
    cmd = ['kubectl', 'get', 'po', '-n', 'kube-system', '-o', 'json']

    try:
        output = check_output(cmd).decode('utf-8')
        result = json.loads(output)
    except CalledProcessError:
        hookenv.log('failed to get kube-system pod status')
        return False
    hookenv.log('Checking system pods status: {}'.format(', '.join(
        '='.join([pod['metadata']['name'], pod['status']['phase']])
        for pod in result['items'])))

    all_pending = all(pod['status']['phase'] == 'Pending'
                      for pod in result['items'])
    if is_state('endpoint.gcp.ready') and all_pending:
        poke_network_unavailable()
        return False

    # All pods must be Running or Evicted (which should re-spawn)
    all_running = all(pod['status']['phase'] == 'Running' or
                      pod['status'].get('reason', '') == 'Evicted'
                      for pod in result['items'])
    return all_running


def poke_network_unavailable():
    """
    Work around https://github.com/kubernetes/kubernetes/issues/44254 by
    manually poking the status into the API server to tell the nodes they have
    a network route.

    This is needed because kubelet sets the NetworkUnavailable flag and expects
    the network plugin to clear it, which only kubenet does. There is some
    discussion about refactoring the affected code but nothing has happened
    in a while.
    """
    cmd = ['kubectl', 'get', 'nodes', '-o', 'json']

    try:
        output = check_output(cmd).decode('utf-8')
        nodes = json.loads(output)['items']
    except CalledProcessError:
        hookenv.log('failed to get kube-system nodes')
        return
    except (KeyError, json.JSONDecodeError) as e:
        hookenv.log('failed to parse kube-system node status '
                    '({}): {}'.format(e, output), hookenv.ERROR)
        return

    for node in nodes:
        node_name = node['metadata']['name']
        url = 'http://localhost:8080/api/v1/nodes/{}/status'.format(node_name)
        with urlopen(url) as response:
            code = response.getcode()
            body = response.read().decode('utf8')
        if code != 200:
            hookenv.log('failed to get node status from {} [{}]: {}'.format(
                url, code, body), hookenv.ERROR)
            return
        try:
            node_info = json.loads(body)
            conditions = node_info['status']['conditions']
            i = [c['type'] for c in conditions].index('NetworkUnavailable')
            if conditions[i]['status'] == 'True':
                hookenv.log('Clearing NetworkUnavailable from {}'.format(
                    node_name))
                conditions[i] = {
                    "type": "NetworkUnavailable",
                    "status": "False",
                    "reason": "RouteCreated",
                    "message": "Manually set through k8s api",
                }
                req = Request(url, method='PUT',
                              data=json.dumps(node_info).encode('utf8'),
                              headers={'Content-Type': 'application/json'})
                with urlopen(req) as response:
                    code = response.getcode()
                    body = response.read().decode('utf8')
                if code not in (200, 201, 202):
                    hookenv.log('failed to update node status [{}]: {}'.format(
                        code, body), hookenv.ERROR)
                    return
        except (json.JSONDecodeError, KeyError):
            hookenv.log('failed to parse node status: {}'.format(body),
                        hookenv.ERROR)
            return


def apiserverVersion():
    cmd = 'kube-apiserver --version'.split()
    version_string = check_output(cmd).decode('utf-8')
    return tuple(int(q) for q in re.findall("[0-9]+", version_string)[:3])


def touch(fname):
    try:
        os.utime(fname, None)
    except OSError:
        open(fname, 'a').close()


def getStorageBackend():
    storage_backend = hookenv.config('storage-backend')
    if storage_backend == 'auto':
        storage_backend = leader_get('auto_storage_backend')
    return storage_backend


@when('leadership.is_leader')
@when_not('leadership.set.cluster_tag')
def create_cluster_tag():
    cluster_tag = 'kubernetes-{}'.format(token_generator().lower())
    leader_set(cluster_tag=cluster_tag)


@when('leadership.set.cluster_tag',
      'kube-control.connected')
@when_not('kubernetes-master.cluster-tag-sent')
def send_cluster_tag():
    cluster_tag = leader_get('cluster_tag')
    kube_control = endpoint_from_flag('kube-control.connected')
    kube_control.set_cluster_tag(cluster_tag)
    set_state('kubernetes-master.cluster-tag-sent')


@when_not('kube-control.connected')
def clear_cluster_tag_sent():
    remove_state('kubernetes-master.cluster-tag-sent')


@when_any('endpoint.aws.joined',
          'endpoint.gcp.joined',
          'endpoint.openstack.joined',
          'endpoint.vsphere.joined',
          'endpoint.azure.joined')
@when_not('kubernetes-master.cloud.ready')
def set_cloud_pending():
    k8s_version = get_version('kube-apiserver')
    k8s_1_11 = k8s_version >= (1, 11)
    k8s_1_12 = k8s_version >= (1, 12)
    vsphere_joined = is_state('endpoint.vsphere.joined')
    azure_joined = is_state('endpoint.azure.joined')
    if (vsphere_joined and not k8s_1_12) or (azure_joined and not k8s_1_11):
        set_state('kubernetes-master.cloud.blocked')
    else:
        remove_state('kubernetes-master.cloud.blocked')
    set_state('kubernetes-master.cloud.pending')


@when_any('endpoint.aws.joined',
          'endpoint.gcp.joined',
          'endpoint.azure.joined')
@when('leadership.set.cluster_tag')
@when_not('kubernetes-master.cloud.request-sent')
def request_integration():
    hookenv.status_set('maintenance', 'requesting cloud integration')
    cluster_tag = leader_get('cluster_tag')
    if is_state('endpoint.aws.joined'):
        cloud = endpoint_from_flag('endpoint.aws.joined')
        cloud.tag_instance({
            'kubernetes.io/cluster/{}'.format(cluster_tag): 'owned',
            'k8s.io/role/master': 'true',
        })
        cloud.tag_instance_security_group({
            'kubernetes.io/cluster/{}'.format(cluster_tag): 'owned',
        })
        cloud.tag_instance_subnet({
            'kubernetes.io/cluster/{}'.format(cluster_tag): 'owned',
        })
        cloud.enable_object_storage_management(['kubernetes-*'])
        cloud.enable_load_balancer_management()
    elif is_state('endpoint.gcp.joined'):
        cloud = endpoint_from_flag('endpoint.gcp.joined')
        cloud.label_instance({
            'k8s-io-cluster-name': cluster_tag,
            'k8s-io-role-master': 'master',
        })
        cloud.enable_object_storage_management()
        cloud.enable_security_management()
    elif is_state('endpoint.azure.joined'):
        cloud = endpoint_from_flag('endpoint.azure.joined')
        cloud.tag_instance({
            'k8s-io-cluster-name': cluster_tag,
            'k8s-io-role-master': 'master',
        })
        cloud.enable_object_storage_management()
        cloud.enable_security_management()
    cloud.enable_instance_inspection()
    cloud.enable_network_management()
    cloud.enable_dns_management()
    cloud.enable_block_storage_management()
    set_state('kubernetes-master.cloud.request-sent')


@when_none('endpoint.aws.joined',
           'endpoint.gcp.joined',
           'endpoint.openstack.joined',
           'endpoint.vsphere.joined',
           'endpoint.azure.joined')
def clear_cloud_flags():
    remove_state('kubernetes-master.cloud.pending')
    remove_state('kubernetes-master.cloud.request-sent')
    remove_state('kubernetes-master.cloud.blocked')
    remove_state('kubernetes-master.cloud.ready')


@when_any('endpoint.aws.ready',
          'endpoint.gcp.ready',
          'endpoint.openstack.ready',
          'endpoint.vsphere.ready',
          'endpoint.azure.ready')
@when_not('kubernetes-master.cloud.blocked',
          'kubernetes-master.cloud.ready')
def cloud_ready():
    if is_state('endpoint.gcp.ready'):
        _write_gcp_snap_config('kube-apiserver')
        _write_gcp_snap_config('kube-controller-manager')
    elif is_state('endpoint.openstack.ready'):
        _write_openstack_snap_config('kube-apiserver')
        _write_openstack_snap_config('kube-controller-manager')
    elif is_state('endpoint.vsphere.ready'):
        _write_vsphere_snap_config('kube-apiserver')
        _write_vsphere_snap_config('kube-controller-manager')
    elif is_state('endpoint.azure.ready'):
        _write_azure_snap_config('kube-apiserver')
        _write_azure_snap_config('kube-controller-manager')
    remove_state('kubernetes-master.cloud.pending')
    set_state('kubernetes-master.cloud.ready')
    remove_state('kubernetes-master.components.started')  # force restart


def _snap_common_path(component):
    return Path('/var/snap/{}/common'.format(component))


def _cloud_config_path(component):
    return _snap_common_path(component) / 'cloud-config.conf'


def _gcp_creds_path(component):
    return _snap_common_path(component) / 'gcp-creds.json'


def _daemon_env_path(component):
    return _snap_common_path(component) / 'environment'


def _cdk_addons_template_path():
    return Path('/snap/cdk-addons/current/templates')


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


def _write_vsphere_snap_config(component):
    # vsphere requires additional cloud config
    vsphere = endpoint_from_flag('endpoint.vsphere.ready')

    # NB: vsphere provider will ask kube-apiserver and -controller-manager to
    # find a uuid from sysfs unless a global config value is set. Our strict
    # snaps cannot read sysfs, so let's do it in the charm. An invalid uuid is
    # not fatal for storage, but it will muddy the logs; try to get it right.
    uuid_file = '/sys/class/dmi/id/product_uuid'
    try:
        with open(uuid_file, 'r') as f:
            uuid = f.read().strip()
    except IOError as err:
        hookenv.log("Unable to read UUID from sysfs: {}".format(err))
        uuid = 'UNKNOWN'

    cloud_config_path = _cloud_config_path(component)
    cloud_config_path.write_text('\n'.join([
        '[Global]',
        'insecure-flag = true',
        'datacenters = "{}"'.format(vsphere.datacenter),
        'vm-uuid = "VMware-{}"'.format(uuid),
        '[VirtualCenter "{}"]'.format(vsphere.vsphere_ip),
        'user = {}'.format(vsphere.user),
        'password = {}'.format(vsphere.password),
        '[Workspace]',
        'server = {}'.format(vsphere.vsphere_ip),
        'datacenter = "{}"'.format(vsphere.datacenter),
        'default-datastore = "{}"'.format(vsphere.datastore),
        'folder = "kubernetes"',
        'resourcepool-path = ""',
        '[Disk]',
        'scsicontrollertype = "pvscsi"',
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
