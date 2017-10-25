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

import os
import random
import shutil
import subprocess
import time

from shlex import split
from subprocess import check_call, check_output
from subprocess import CalledProcessError
from socket import gethostname

from charms import layer
from charms.layer import snap
from charms.reactive import hook
from charms.reactive import set_state, remove_state, is_state
from charms.reactive import when, when_any, when_not

from charms.kubernetes.common import get_version
from charms.kubernetes.flagmanager import FlagManager

from charms.reactive.helpers import data_changed, any_file_changed
from charms.templating.jinja2 import render

from charmhelpers.core import hookenv
from charmhelpers.core.host import service_stop, service_restart
from charmhelpers.contrib.charmsupport import nrpe

# Override the default nagios shortname regex to allow periods, which we
# need because our bin names contain them (e.g. 'snap.foo.daemon'). The
# default regex in charmhelpers doesn't allow periods, but nagios itself does.
nrpe.Check.shortname_re = '[\.A-Za-z0-9-_]+$'

kubeconfig_path = '/root/cdk/kubeconfig'
kubeproxyconfig_path = '/root/cdk/kubeproxyconfig'
kubeclientconfig_path = '/root/.kube/config'

os.environ['PATH'] += os.pathsep + os.path.join(os.sep, 'snap', 'bin')


@hook('upgrade-charm')
def upgrade_charm():
    # Trigger removal of PPA docker installation if it was previously set.
    set_state('config.changed.install_from_upstream')
    hookenv.atexit(remove_state, 'config.changed.install_from_upstream')

    cleanup_pre_snap_services()
    check_resources_for_upgrade_needed()

    # Remove gpu.enabled state so we can reconfigure gpu-related kubelet flags,
    # since they can differ between k8s versions
    remove_state('kubernetes-worker.gpu.enabled')
    kubelet_opts = FlagManager('kubelet')
    kubelet_opts.destroy('feature-gates')
    kubelet_opts.destroy('experimental-nvidia-gpus')

    remove_state('kubernetes-worker.cni-plugins.installed')
    remove_state('kubernetes-worker.config.created')
    remove_state('kubernetes-worker.ingress.available')
    set_state('kubernetes-worker.restart-needed')


def check_resources_for_upgrade_needed():
    hookenv.status_set('maintenance', 'Checking resources')
    resources = ['kubectl', 'kubelet', 'kube-proxy']
    paths = [hookenv.resource_get(resource) for resource in resources]
    if any_file_changed(paths):
        set_upgrade_needed()


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

    # cleanup old flagmanagers
    FlagManager('kubelet').destroy_all()
    FlagManager('kube-proxy').destroy_all()


@when('config.changed.channel')
def channel_changed():
    set_upgrade_needed()


@when('kubernetes-worker.snaps.upgrade-needed')
@when_not('kubernetes-worker.snaps.upgrade-specified')
def upgrade_needed_status():
    msg = 'Needs manual upgrade, run the upgrade action'
    hookenv.status_set('blocked', msg)


@when('kubernetes-worker.snaps.upgrade-specified')
def install_snaps():
    check_resources_for_upgrade_needed()
    channel = hookenv.config('channel')
    hookenv.status_set('maintenance', 'Installing kubectl snap')
    snap.install('kubectl', channel=channel, classic=True)
    hookenv.status_set('maintenance', 'Installing kubelet snap')
    snap.install('kubelet', channel=channel, classic=True)
    hookenv.status_set('maintenance', 'Installing kube-proxy snap')
    snap.install('kube-proxy', channel=channel, classic=True)
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
            kubectl('delete', 'node', gethostname())
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
@when_not('kube-control.dns.available')
def notify_user_transient_status():
    ''' Notify to the user we are in a transient state and the application
    is still converging. Potentially remotely, or we may be in a detached loop
    wait state '''

    # During deployment the worker has to start kubelet without cluster dns
    # configured. If this is the first unit online in a service pool waiting
    # to self host the dns pod, and configure itself to query the dns service
    # declared in the kube-system namespace

    hookenv.status_set('waiting', 'Waiting for cluster DNS.')


@when('kubernetes-worker.snaps.installed',
      'kube-control.dns.available')
@when_not('kubernetes-worker.snaps.upgrade-needed')
def charm_status(kube_control):
    '''Update the status message with the current status of kubelet.'''
    update_kubelet_status()


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


@when('certificates.available')
def send_data(tls):
    '''Send the data that is required to create a server certificate for
    this server.'''
    # Use the public ip of this unit as the Common Name for the certificate.
    common_name = hookenv.unit_public_ip()

    # Create SANs that the tls layer will add to the server cert.
    sans = [
        hookenv.unit_public_ip(),
        hookenv.unit_private_ip(),
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
def start_worker(kube_api, kube_control, auth_control, cni):
    ''' Start kubelet using the provided API and DNS info.'''
    servers = get_kube_api_servers(kube_api)
    # Note that the DNS server doesn't necessarily exist at this point. We know
    # what its IP will eventually be, though, so we can go ahead and configure
    # kubelet with that info. This ensures that early pods are configured with
    # the correct DNS even though the server isn't ready yet.

    dns = kube_control.get_dns()
    cluster_cidr = cni.get_config()['cidr']

    if cluster_cidr is None:
        hookenv.log('Waiting for cluster cidr.')
        return

    nodeuser = 'system:node:{}'.format(gethostname())
    creds = kube_control.get_auth_credentials(nodeuser)
    data_changed('kube-control.creds', creds)

    # set --allow-privileged flag for kubelet
    set_privileged()

    create_config(random.choice(servers), creds)
    configure_worker_services(servers, dns, cluster_cidr)
    set_state('kubernetes-worker.config.created')
    restart_unit_services()
    update_kubelet_status()
    apply_node_labels()
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
    ''' If configuration has ingress RC enabled, launch the ingress load
    balancer and default http backend. Otherwise attempt deletion. '''
    config = hookenv.config()
    # If ingress is enabled, launch the ingress controller
    if config.get('ingress'):
        launch_default_ingress_controller()
    else:
        hookenv.log('Deleting the http backend and ingress.')
        kubectl_manifest('delete',
                         '/root/cdk/addons/default-http-backend.yaml')
        kubectl_manifest('delete',
                         '/root/cdk/addons/ingress-replication-controller.yaml')  # noqa
        hookenv.close_port(80)
        hookenv.close_port(443)


@when('kubernetes-worker.ingress.available')
def scale_ingress_controller():
    ''' Scale the number of ingress controller replicas to match the number of
    nodes. '''
    try:
        output = kubectl('get', 'nodes', '-o', 'name')
        count = len(output.splitlines())
        kubectl('scale', '--replicas=%d' % count, 'rc/nginx-ingress-controller')  # noqa
    except CalledProcessError:
        hookenv.log('Failed to scale ingress controllers. Will attempt again next update.')  # noqa


@when('config.changed.labels', 'kubernetes-worker.config.created')
def apply_node_labels():
    ''' Parse the labels configuration option and apply the labels to the node.
    '''
    # scrub and try to format an array from the configuration option
    config = hookenv.config()
    user_labels = _parse_labels(config.get('labels'))

    # For diffing sake, iterate the previous label set
    if config.previous('labels'):
        previous_labels = _parse_labels(config.previous('labels'))
        hookenv.log('previous labels: {}'.format(previous_labels))
    else:
        # this handles first time run if there is no previous labels config
        previous_labels = _parse_labels("")

    # Calculate label removal
    for label in previous_labels:
        if label not in user_labels:
            hookenv.log('Deleting node label {}'.format(label))
            _apply_node_label(label, delete=True)
        # if the label is in user labels we do nothing here, it will get set
        # during the atomic update below.

    # Atomically set a label
    for label in user_labels:
        _apply_node_label(label, overwrite=True)


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


def configure_worker_services(api_servers, dns, cluster_cidr):
    ''' Add remaining flags for the worker services and configure snaps to use
    them '''
    layer_options = layer.options('tls-client')
    ca_cert_path = layer_options.get('ca_certificate_path')
    server_cert_path = layer_options.get('server_certificate_path')
    server_key_path = layer_options.get('server_key_path')

    kubelet_opts = FlagManager('kubelet')
    kubelet_opts.add('require-kubeconfig', 'true')
    kubelet_opts.add('kubeconfig', kubeconfig_path)
    kubelet_opts.add('network-plugin', 'cni')
    kubelet_opts.add('v', '0')
    kubelet_opts.add('address', '0.0.0.0')
    kubelet_opts.add('port', '10250')
    kubelet_opts.add('cluster-dns', dns['sdn-ip'])
    kubelet_opts.add('cluster-domain', dns['domain'])
    kubelet_opts.add('anonymous-auth', 'false')
    kubelet_opts.add('client-ca-file', ca_cert_path)
    kubelet_opts.add('tls-cert-file', server_cert_path)
    kubelet_opts.add('tls-private-key-file', server_key_path)
    kubelet_opts.add('logtostderr', 'true')
    kubelet_opts.add('fail-swap-on', 'false')

    kube_proxy_opts = FlagManager('kube-proxy')
    kube_proxy_opts.add('cluster-cidr', cluster_cidr)
    kube_proxy_opts.add('kubeconfig', kubeproxyconfig_path)
    kube_proxy_opts.add('logtostderr', 'true')
    kube_proxy_opts.add('v', '0')
    kube_proxy_opts.add('master', random.choice(api_servers), strict=True)

    if b'lxc' in check_output('virt-what', shell=True):
        kube_proxy_opts.add('conntrack-max-per-core', '0')

    cmd = ['snap', 'set', 'kubelet'] + kubelet_opts.to_s().split(' ')
    check_call(cmd)
    cmd = ['snap', 'set', 'kube-proxy'] + kube_proxy_opts.to_s().split(' ')
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


def launch_default_ingress_controller():
    ''' Launch the Kubernetes ingress controller & default backend (404) '''
    context = {}
    context['arch'] = arch()
    addon_path = '/root/cdk/addons/{}'

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

    # Render the ingress replication controller manifest
    context['ingress_image'] = \
        "gcr.io/google_containers/nginx-ingress-controller:0.9.0-beta.13"
    if arch() == 's390x':
        context['ingress_image'] = \
            "docker.io/cdkbot/nginx-ingress-controller-s390x:0.9.0-beta.13"
    manifest = addon_path.format('ingress-replication-controller.yaml')
    render('ingress-replication-controller.yaml', manifest, context)
    hookenv.log('Creating the ingress replication controller.')
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
    ''' Runs kubectl with the given args. Returns True if succesful, False if
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
    """Update the allow-privileged flag for kubelet.

    """
    privileged = hookenv.config('allow-privileged')
    if privileged == 'auto':
        gpu_enabled = is_state('kubernetes-worker.gpu.enabled')
        privileged = 'true' if gpu_enabled else 'false'

    flag = 'allow-privileged'
    hookenv.log('Setting {}={}'.format(flag, privileged))

    kubelet_opts = FlagManager('kubelet')
    kubelet_opts.add(flag, privileged)

    if privileged == 'true':
        set_state('kubernetes-worker.privileged')
    else:
        remove_state('kubernetes-worker.privileged')


@when('config.changed.allow-privileged')
@when('kubernetes-worker.config.created')
def on_config_allow_privileged_change():
    """React to changed 'allow-privileged' config value.

    """
    set_state('kubernetes-worker.restart-needed')
    remove_state('config.changed.allow-privileged')


@when('cuda.installed')
@when('kubernetes-worker.config.created')
@when_not('kubernetes-worker.gpu.enabled')
def enable_gpu():
    """Enable GPU usage on this node.

    """
    config = hookenv.config()
    if config['allow-privileged'] == "false":
        hookenv.status_set(
            'active',
            'GPUs available. Set allow-privileged="auto" to enable.'
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

    kubelet_opts = FlagManager('kubelet')
    if get_version('kubelet') < (1, 6):
        hookenv.log('Adding --experimental-nvidia-gpus=1 to kubelet')
        kubelet_opts.add('experimental-nvidia-gpus', '1')
    else:
        hookenv.log('Adding --feature-gates=Accelerators=true to kubelet')
        kubelet_opts.add('feature-gates', 'Accelerators=true')

    # Apply node labels
    _apply_node_label('gpu=true', overwrite=True)
    _apply_node_label('cuda=true', overwrite=True)

    set_state('kubernetes-worker.gpu.enabled')
    set_state('kubernetes-worker.restart-needed')


@when('kubernetes-worker.gpu.enabled')
@when_not('kubernetes-worker.privileged')
@when_not('kubernetes-worker.restart-needed')
def disable_gpu():
    """Disable GPU usage on this node.

    This handler fires when we're running in gpu mode, and then the operator
    sets allow-privileged="false". Since we can no longer run privileged
    containers, we need to disable gpu mode.

    """
    hookenv.log('Disabling gpu mode')

    kubelet_opts = FlagManager('kubelet')
    if get_version('kubelet') < (1, 6):
        kubelet_opts.destroy('experimental-nvidia-gpus')
    else:
        kubelet_opts.remove('feature-gates', 'Accelerators=true')

    # Remove node labels
    _apply_node_label('gpu', delete=True)
    _apply_node_label('cuda', delete=True)

    remove_state('kubernetes-worker.gpu.enabled')
    set_state('kubernetes-worker.restart-needed')


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
    nodeuser = 'system:node:{}'.format(gethostname())
    kube_control.set_auth_request(nodeuser)


@when('kube-control.connected')
def catch_change_in_creds(kube_control):
    """Request a service restart in case credential updates were detected."""
    nodeuser = 'system:node:{}'.format(gethostname())
    creds = kube_control.get_auth_credentials(nodeuser)
    if creds \
            and data_changed('kube-control.creds', creds) \
            and creds['user'] == nodeuser:
        set_state('worker.auth.bootstrapped')
        set_state('kubernetes-worker.restart-needed')


@when_not('kube-control.connected')
def missing_kube_control():
    """Inform the operator they need to add the kube-control relation.

    If deploying via bundle this won't happen, but if operator is upgrading a
    a charm in a deployment that pre-dates the kube-control relation, it'll be
    missing.

    """
    hookenv.status_set(
        'blocked',
        'Relate {}:kube-control kubernetes-master:kube-control'.format(
            hookenv.service_name()))


def _systemctl_is_active(application):
    ''' Poll systemctl to determine if the application is running '''
    cmd = ['systemctl', 'is-active', application]
    try:
        raw = check_output(cmd)
        return b'active' in raw
    except Exception:
        return False


class ApplyNodeLabelFailed(Exception):
    pass


def _apply_node_label(label, delete=False, overwrite=False):
    ''' Invoke kubectl to apply node label changes '''

    hostname = gethostname()
    # TODO: Make this part of the kubectl calls instead of a special string
    cmd_base = 'kubectl --kubeconfig={0} label node {1} {2}'

    if delete is True:
        label_key = label.split('=')[0]
        cmd = cmd_base.format(kubeconfig_path, hostname, label_key)
        cmd = cmd + '-'
    else:
        cmd = cmd_base.format(kubeconfig_path, hostname, label)
        if overwrite:
            cmd = '{} --overwrite'.format(cmd)
    cmd = cmd.split()

    deadline = time.time() + 60
    while time.time() < deadline:
        code = subprocess.call(cmd)
        if code == 0:
            break
        hookenv.log('Failed to apply label %s, exit code %d. Will retry.' % (
            label, code))
        time.sleep(1)
    else:
        msg = 'Failed to apply label %s' % label
        raise ApplyNodeLabelFailed(msg)


def _parse_labels(labels):
    ''' Parse labels from a key=value string separated by space.'''
    label_array = labels.split(' ')
    sanitized_labels = []
    for item in label_array:
        if '=' in item:
            sanitized_labels.append(item)
        else:
            hookenv.log('Skipping malformed option: {}'.format(item))
    return sanitized_labels
