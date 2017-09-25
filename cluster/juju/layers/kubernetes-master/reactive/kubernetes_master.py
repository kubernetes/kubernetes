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
import os
import re
import random
import shutil
import socket
import string
import json
import ipaddress

import charms.leadership

from shlex import split
from subprocess import check_call
from subprocess import check_output
from subprocess import CalledProcessError

from charms import layer
from charms.layer import snap
from charms.reactive import hook
from charms.reactive import remove_state
from charms.reactive import set_state
from charms.reactive import is_state
from charms.reactive import when, when_any, when_not, when_all
from charms.reactive.helpers import data_changed, any_file_changed
from charms.kubernetes.common import get_version
from charms.kubernetes.common import retry
from charms.kubernetes.flagmanager import FlagManager

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

os.environ['PATH'] += os.pathsep + os.path.join(os.sep, 'snap', 'bin')


def service_cidr():
    ''' Return the charm's service-cidr config '''
    db = unitdata.kv()
    frozen_cidr = db.get('kubernetes-master.service-cidr')
    return frozen_cidr or hookenv.config('service-cidr')


def freeze_service_cidr():
    ''' Freeze the service CIDR. Once the apiserver has started, we can no
    longer safely change this value. '''
    db = unitdata.kv()
    db.set('kubernetes-master.service-cidr', service_cidr())


@hook('upgrade-charm')
def reset_states_for_delivery():
    '''An upgrade charm event was triggered by Juju, react to that here.'''
    migrate_from_pre_snaps()
    install_snaps()
    set_state('reconfigure.authentication.setup')
    remove_state('authentication.setup')


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

    # clear the flag managers
    FlagManager('kube-apiserver').destroy_all()
    FlagManager('kube-controller-manager').destroy_all()
    FlagManager('kube-scheduler').destroy_all()


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
    set_state('kubernetes-master.snaps.installed')
    remove_state('kubernetes-master.components.started')


@when('config.changed.channel')
def channel_changed():
    install_snaps()


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
    setup_basic_auth(password, "admin", "admin")
    set_state('reconfigure.authentication.setup')
    remove_state('authentication.setup')
    set_state('client.password.initialised')


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
    api_opts = FlagManager('kube-apiserver')
    controller_opts = FlagManager('kube-controller-manager')

    service_key = '/root/cdk/serviceaccount.key'
    basic_auth = '/root/cdk/basic_auth.csv'
    known_tokens = '/root/cdk/known_tokens.csv'

    api_opts.add('basic-auth-file', basic_auth)
    api_opts.add('token-auth-file', known_tokens)
    hookenv.status_set('maintenance', 'Rendering authentication templates.')

    keys = [service_key, basic_auth, known_tokens]
    # Try first to fetch data from an old leadership broadcast.
    if not get_keys_from_leader(keys) \
            or is_state('reconfigure.authentication.setup'):
        last_pass = get_password('basic_auth.csv', 'admin')
        setup_basic_auth(last_pass, 'admin', 'admin')

        if not os.path.isfile(known_tokens):
            setup_tokens(None, 'admin', 'admin')
            setup_tokens(None, 'kubelet', 'kubelet')
            setup_tokens(None, 'kube_proxy', 'kube_proxy')

        # Generate the default service account token key
        os.makedirs('/root/cdk', exist_ok=True)
        if not os.path.isfile(service_key):
            cmd = ['openssl', 'genrsa', '-out', service_key,
                   '2048']
            check_call(cmd)
        remove_state('reconfigure.authentication.setup')

    api_opts.add('service-account-key-file', service_key)
    controller_opts.add('service-account-private-key-file', service_key)

    # read service account key for syndication
    leader_data = {}
    for f in [known_tokens, basic_auth, service_key]:
        with open(f, 'r') as fp:
            leader_data[f] = fp.read()

    # this is slightly opaque, but we are sending file contents under its file
    # path as a key.
    # eg:
    # {'/root/cdk/serviceaccount.key': 'RSA:2471731...'}
    charms.leadership.leader_set(leader_data)
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
    api_opts = FlagManager('kube-apiserver')
    api_opts.add('basic-auth-file', basic_auth)
    api_opts.add('token-auth-file', known_tokens)
    api_opts.add('service-account-key-file', service_key)

    controller_opts = FlagManager('kube-controller-manager')
    controller_opts.add('service-account-private-key-file', service_key)

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
            contents = charms.leadership.leader_get(k)
            # Default to logging the warning and wait for leader data to be set
            if contents is None:
                msg = "Waiting on leaders crypto keys."
                hookenv.status_set('waiting', msg)
                hookenv.log('Missing content for file {}'.format(k))
                return False
            # Write out the file and move on to the next item
            with open(k, 'w+') as fp:
                fp.write(contents)

    return True


@when('kubernetes-master.snaps.installed')
def set_app_version():
    ''' Declare the application version to juju '''
    version = check_output(['kube-apiserver', '--version'])
    hookenv.application_version_set(version.split(b' v')[-1].rstrip())


@when('cdk-addons.configured', 'kube-api-endpoint.available',
      'kube-control.connected')
def idle_status(kube_api, kube_control):
    ''' Signal at the end of the run that we are running. '''
    if not all_kube_system_pods_running():
        hookenv.status_set('waiting', 'Waiting for kube-system pods to start')
    elif hookenv.config('service-cidr') != service_cidr():
        msg = 'WARN: cannot change service-cidr, still using ' + service_cidr()
        hookenv.status_set('active', msg)
    else:
        # All services should be up and running at this point. Double-check...
        failing_services = master_services_down()
        if len(failing_services) == 0:
            hookenv.status_set('active', 'Kubernetes master running.')
        else:
            msg = 'Stopped services: {}'.format(','.join(failing_services))
            hookenv.status_set('blocked', msg)


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
@when_not('kubernetes-master.components.started')
def start_master(etcd):
    '''Run the Kubernetes master components.'''
    hookenv.status_set('maintenance',
                       'Configuring the Kubernetes master services.')
    freeze_service_cidr()
    if not etcd.get_connection_string():
        # etcd is not returning a connection string. This hapens when
        # the master unit disconnects from etcd and is ready to terminate.
        # No point in trying to start master services and fail. Just return.
        return

    # TODO: Make sure below relation is handled on change
    # https://github.com/kubernetes/kubernetes/issues/43461
    handle_etcd_relation(etcd)

    # Add CLI options to all components
    configure_apiserver()
    configure_controller_manager()
    configure_scheduler()

    hookenv.open_port(6443)


@when('etcd.available')
def etcd_data_change(etcd):
    ''' Etcd scale events block master reconfiguration due to the
        kubernetes-master.components.started state. We need a way to
        handle these events consistenly only when the number of etcd
        units has actually changed '''

    # key off of the connection string
    connection_string = etcd.get_connection_string()

    # If the connection string changes, remove the started state to trigger
    # handling of the master components
    if data_changed('etcd-connect', connection_string):
        remove_state('kubernetes-master.components.started')


@when('kube-control.connected')
@when('cdk-addons.configured')
def send_cluster_dns_detail(kube_control):
    ''' Send cluster DNS info '''
    # Note that the DNS server doesn't necessarily exist at this point. We know
    # where we're going to put it, though, so let's send the info anyway.
    dns_ip = get_dns_ip()
    kube_control.set_dns(53, hookenv.config('dns_domain'), dns_ip)


@when('kube-control.auth.requested')
@when('authentication.setup')
@when('leadership.is_leader')
def send_tokens(kube_control):
    """Send the tokens to the workers."""
    kubelet_token = get_token('kubelet')
    proxy_token = get_token('kube_proxy')
    admin_token = get_token('admin')

    # Send the data
    requests = kube_control.auth_user()
    for request in requests:
        kube_control.sign_auth_request(request[0], kubelet_token,
                                       proxy_token, admin_token)


@when_not('kube-control.connected')
def missing_kube_control():
    """Inform the operator master is waiting for a relation to workers.

    If deploying via bundle this won't happen, but if operator is upgrading a
    a charm in a deployment that pre-dates the kube-control relation, it'll be
    missing.

    """
    hookenv.status_set('blocked', 'Waiting for workers.')


@when('kube-api-endpoint.available')
def push_service_data(kube_api):
    ''' Send configuration to the load balancer, and close access to the
    public interface '''
    kube_api.configure(port=6443)


@when('certificates.available')
def send_data(tls):
    '''Send the data that is required to create a server certificate for
    this server.'''
    # Use the public ip of this unit as the Common Name for the certificate.
    common_name = hookenv.unit_public_ip()

    # Get the SDN gateway based on the cidr address.
    kubernetes_service_ip = get_kubernetes_service_ip()

    domain = hookenv.config('dns_domain')
    # Create SANs that the tls layer will add to the server cert.
    sans = [
        hookenv.unit_public_ip(),
        hookenv.unit_private_ip(),
        socket.gethostname(),
        kubernetes_service_ip,
        'kubernetes',
        'kubernetes.{0}'.format(domain),
        'kubernetes.default',
        'kubernetes.default.svc',
        'kubernetes.default.svc.{0}'.format(domain)
    ]
    # Create a path safe name by removing path characters from the unit name.
    certificate_name = hookenv.local_unit().replace('/', '_')
    # Request a server cert with this information.
    tls.request_server_cert(common_name, sans, certificate_name)


@when('kubernetes-master.components.started')
def configure_cdk_addons():
    ''' Configure CDK addons '''
    remove_state('cdk-addons.configured')
    dbEnabled = str(hookenv.config('enable-dashboard-addons')).lower()
    args = [
        'arch=' + arch(),
        'dns-ip=' + get_dns_ip(),
        'dns-domain=' + hookenv.config('dns_domain'),
        'enable-dashboard=' + dbEnabled
    ]
    check_call(['snap', 'set', 'cdk-addons'] + args)
    if not addons_ready():
        hookenv.status_set('waiting', 'Waiting to retry addon deployment')
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
    except:
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
    privileged = hookenv.config('allow-privileged')
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


@when('config.changed.api-extra-args')
@when('kubernetes-master.components.started')
def on_config_api_extra_args_change():
    configure_apiserver()


@when('kube-control.gpu.available')
@when('kubernetes-master.components.started')
@when_not('kubernetes-master.gpu.enabled')
def on_gpu_available(kube_control):
    """The remote side (kubernetes-worker) is gpu-enabled.

    We need to run in privileged mode.

    """
    config = hookenv.config()
    if config['allow-privileged'] == "false":
        hookenv.status_set(
            'active',
            'GPUs available. Set allow-privileged="auto" to enable.'
        )
        return

    remove_state('kubernetes-master.components.started')
    set_state('kubernetes-master.gpu.enabled')


@when('kubernetes-master.gpu.enabled')
@when_not('kubernetes-master.privileged')
def disable_gpu_mode():
    """We were in gpu mode, but the operator has set allow-privileged="false",
    so we can't run in gpu mode anymore.

    """
    remove_state('kubernetes-master.gpu.enabled')


@hook('stop')
def shutdown():
    """ Stop the kubernetes master services

    """
    service_stop('snap.kube-apiserver.daemon')
    service_stop('snap.kube-controller-manager.daemon')
    service_stop('snap.kube-scheduler.daemon')


@when('kube-apiserver.do-restart')
def restart_apiserver():
    prev_state, prev_msg = hookenv.status_get()
    hookenv.status_set('maintenance', 'Restarting kube-apiserver')
    host.service_restart('snap.kube-apiserver.daemon')
    hookenv.status_set(prev_state, prev_msg)
    remove_state('kube-apiserver.do-restart')
    set_state('kube-apiserver.started')


@when('kube-controller-manager.do-restart')
def restart_controller_manager():
    prev_state, prev_msg = hookenv.status_get()
    hookenv.status_set('maintenance', 'Restarting kube-controller-manager')
    host.service_restart('snap.kube-controller-manager.daemon')
    hookenv.status_set(prev_state, prev_msg)
    remove_state('kube-controller-manager.do-restart')
    set_state('kube-controller-manager.started')


@when('kube-scheduler.do-restart')
def restart_scheduler():
    prev_state, prev_msg = hookenv.status_get()
    hookenv.status_set('maintenance', 'Restarting kube-scheduler')
    host.service_restart('snap.kube-scheduler.daemon')
    hookenv.status_set(prev_state, prev_msg)
    remove_state('kube-scheduler.do-restart')
    set_state('kube-scheduler.started')


@when_all('kube-apiserver.started',
          'kube-controller-manager.started',
          'kube-scheduler.started')
@when_not('kubernetes-master.components.started')
def componenets_started():
    set_state('kubernetes-master.components.started')


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
    '''Get an IP address for the DNS server on the provided cidr.'''
    interface = ipaddress.IPv4Interface(service_cidr())
    # Add .10 at the end of the network
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
    connection_string = reldata.get_connection_string()
    # Define where the etcd tls files will be kept.
    etcd_dir = '/root/cdk/etcd'
    # Create paths to the etcd client ca, key, and cert file locations.
    ca = os.path.join(etcd_dir, 'client-ca.pem')
    key = os.path.join(etcd_dir, 'client-key.pem')
    cert = os.path.join(etcd_dir, 'client-cert.pem')

    # Save the client credentials (in relation data) to the paths provided.
    reldata.save_client_credentials(key, cert, ca)

    api_opts = FlagManager('kube-apiserver')

    # Never use stale data, always prefer whats coming in during context
    # building. if its stale, its because whats in unitdata is stale
    data = api_opts.data
    if data.get('etcd-servers-strict') or data.get('etcd-servers'):
        api_opts.destroy('etcd-cafile')
        api_opts.destroy('etcd-keyfile')
        api_opts.destroy('etcd-certfile')
        api_opts.destroy('etcd-servers', strict=True)
        api_opts.destroy('etcd-servers')

    # Set the apiserver flags in the options manager
    api_opts.add('etcd-cafile', ca)
    api_opts.add('etcd-keyfile', key)
    api_opts.add('etcd-certfile', cert)
    api_opts.add('etcd-servers', connection_string, strict=True)


def get_config_args():
    db = unitdata.kv()
    old_config_args = db.get('api-extra-args', [])
    # We have to convert them to tuples becuase we use sets
    old_config_args = [tuple(i) for i in old_config_args]
    new_config_args = []
    new_config_arg_names = []
    for arg in hookenv.config().get('api-extra-args', '').split():
        new_config_arg_names.append(arg.split('=', 1)[0])
        if len(arg.split('=', 1)) == 1:  # handle flags ie. --profiling
            new_config_args.append(tuple([arg, 'true']))
        else:
            new_config_args.append(tuple(arg.split('=', 1)))

    hookenv.log('Handling "api-extra-args" option.')
    hookenv.log('Old arguments: {}'.format(old_config_args))
    hookenv.log('New arguments: {}'.format(new_config_args))
    if set(new_config_args) == set(old_config_args):
        return (new_config_args, [])
    # Store new args
    db.set('api-extra-args', new_config_args)
    to_add = set(new_config_args)
    to_remove = set(old_config_args) - set(new_config_args)
    # Extract option names only
    to_remove = [i[0] for i in to_remove if i[0] not in new_config_arg_names]
    return (to_add, to_remove)


def configure_apiserver():
    # TODO: investigate if it's possible to use config file to store args
    # https://github.com/juju-solutions/bundle-canonical-kubernetes/issues/315
    # Handle api-extra-args config option
    to_add, to_remove = get_config_args()

    api_opts = FlagManager('kube-apiserver')

    # Remove arguments that are no longer provided as config option
    # this allows them to be reverted to charm defaults
    for arg in to_remove:
        hookenv.log('Removing option: {}'.format(arg))
        api_opts.destroy(arg)
        # We need to "unset" options by settig their value to "null" string
        cmd = ['snap', 'set', 'kube-apiserver', '{}=null'.format(arg)]
        check_call(cmd)

    # Get the tls paths from the layer data.
    layer_options = layer.options('tls-client')
    ca_cert_path = layer_options.get('ca_certificate_path')
    client_cert_path = layer_options.get('client_certificate_path')
    client_key_path = layer_options.get('client_key_path')
    server_cert_path = layer_options.get('server_certificate_path')
    server_key_path = layer_options.get('server_key_path')

    if is_privileged():
        api_opts.add('allow-privileged', 'true', strict=True)
        set_state('kubernetes-master.privileged')
    else:
        api_opts.add('allow-privileged', 'false', strict=True)
        remove_state('kubernetes-master.privileged')

    # Handle static options for now
    api_opts.add('service-cluster-ip-range', service_cidr())
    api_opts.add('min-request-timeout', '300')
    api_opts.add('v', '4')
    api_opts.add('tls-cert-file', server_cert_path)
    api_opts.add('tls-private-key-file', server_key_path)
    api_opts.add('kubelet-certificate-authority', ca_cert_path)
    api_opts.add('kubelet-client-certificate', client_cert_path)
    api_opts.add('kubelet-client-key', client_key_path)
    api_opts.add('logtostderr', 'true')
    api_opts.add('insecure-bind-address', '127.0.0.1')
    api_opts.add('insecure-port', '8080')
    api_opts.add('storage-backend', 'etcd2')  # FIXME: add etcd3 support

    admission_control = [
        'Initializers',
        'NamespaceLifecycle',
        'LimitRanger',
        'ServiceAccount',
        'ResourceQuota',
        'DefaultTolerationSeconds'
    ]

    if get_version('kube-apiserver') < (1, 6):
        hookenv.log('Removing DefaultTolerationSeconds from admission-control')
        admission_control.remove('DefaultTolerationSeconds')
    if get_version('kube-apiserver') < (1, 7):
        hookenv.log('Removing Initializers from admission-control')
        admission_control.remove('Initializers')
    api_opts.add('admission-control', ','.join(admission_control), strict=True)

    # Add operator-provided arguments, this allows operators
    # to override defaults
    for arg in to_add:
        hookenv.log('Adding option: {} {}'.format(arg[0], arg[1]))
        # Make sure old value is gone
        api_opts.destroy(arg[0])
        api_opts.add(arg[0], arg[1])

    cmd = ['snap', 'set', 'kube-apiserver'] + api_opts.to_s().split(' ')
    check_call(cmd)
    set_state('kube-apiserver.do-restart')


def configure_controller_manager():
    controller_opts = FlagManager('kube-controller-manager')

    # Get the tls paths from the layer data.
    layer_options = layer.options('tls-client')
    ca_cert_path = layer_options.get('ca_certificate_path')

    # Default to 3 minute resync. TODO: Make this configureable?
    controller_opts.add('min-resync-period', '3m')
    controller_opts.add('v', '2')
    controller_opts.add('root-ca-file', ca_cert_path)
    controller_opts.add('logtostderr', 'true')
    controller_opts.add('master', 'http://127.0.0.1:8080')

    cmd = (
        ['snap', 'set', 'kube-controller-manager'] +
        controller_opts.to_s().split(' ')
    )
    check_call(cmd)
    set_state('kube-controller-manager.do-restart')


def configure_scheduler():
    scheduler_opts = FlagManager('kube-scheduler')

    scheduler_opts.add('v', '2')
    scheduler_opts.add('logtostderr', 'true')
    scheduler_opts.add('master', 'http://127.0.0.1:8080')

    cmd = ['snap', 'set', 'kube-scheduler'] + scheduler_opts.to_s().split(' ')
    check_call(cmd)
    set_state('kube-scheduler.do-restart')


def setup_basic_auth(password=None, username='admin', uid='admin'):
    '''Create the htacces file and the tokens.'''
    root_cdk = '/root/cdk'
    if not os.path.isdir(root_cdk):
        os.makedirs(root_cdk)
    htaccess = os.path.join(root_cdk, 'basic_auth.csv')
    if not password:
        password = token_generator()
    with open(htaccess, 'w') as stream:
        stream.write('{0},{1},{2}'.format(password, username, uid))


def setup_tokens(token, username, user):
    '''Create a token file for kubernetes authentication.'''
    root_cdk = '/root/cdk'
    if not os.path.isdir(root_cdk):
        os.makedirs(root_cdk)
    known_tokens = os.path.join(root_cdk, 'known_tokens.csv')
    if not token:
        token = token_generator()
    with open(known_tokens, 'a') as stream:
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
    db = unitdata.kv()
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
    except CalledProcessError:
        hookenv.log('failed to get kube-system pod status')
        return False

    result = json.loads(output)
    for pod in result['items']:
        status = pod['status']['phase']
        if status != 'Running':
            return False

    return True


def apiserverVersion():
    cmd = 'kube-apiserver --version'.split()
    version_string = check_output(cmd).decode('utf-8')
    return tuple(int(q) for q in re.findall("[0-9]+", version_string)[:3])
