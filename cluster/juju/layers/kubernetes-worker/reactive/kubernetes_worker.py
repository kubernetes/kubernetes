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

from shlex import split
from subprocess import call, check_call, check_output
from subprocess import CalledProcessError
from socket import gethostname

from charms import layer
from charms.reactive import hook
from charms.reactive import set_state, remove_state
from charms.reactive import when, when_any, when_not
from charms.reactive.helpers import data_changed
from charms.kubernetes.flagmanager import FlagManager
from charms.templating.jinja2 import render

from charmhelpers.core import hookenv
from charmhelpers.core.host import service_stop
from charmhelpers.contrib.charmsupport import nrpe


kubeconfig_path = '/srv/kubernetes/config'


@hook('upgrade-charm')
def remove_installed_state():
    remove_state('kubernetes-worker.components.installed')


@hook('stop')
def shutdown():
    ''' When this unit is destroyed:
        - delete the current node
        - stop the kubelet service
        - stop the kube-proxy service
        - remove the 'kubernetes-worker.components.installed' state
    '''
    kubectl('delete', 'node', gethostname())
    service_stop('kubelet')
    service_stop('kube-proxy')
    remove_state('kubernetes-worker.components.installed')


@when('docker.available')
@when_not('kubernetes-worker.components.installed')
def install_kubernetes_components():
    ''' Unpack the kubernetes worker binaries '''
    charm_dir = os.getenv('CHARM_DIR')

    # Get the resource via resource_get
    try:
        archive = hookenv.resource_get('kubernetes')
    except Exception:
        message = 'Error fetching the kubernetes resource.'
        hookenv.log(message)
        hookenv.status_set('blocked', message)
        return

    if not archive:
        hookenv.log('Missing kubernetes resource.')
        hookenv.status_set('blocked', 'Missing kubernetes resource.')
        return

    # Handle null resource publication, we check if filesize < 1mb
    filesize = os.stat(archive).st_size
    if filesize < 1000000:
        hookenv.status_set('blocked', 'Incomplete kubernetes resource.')
        return

    hookenv.status_set('maintenance', 'Unpacking kubernetes resource.')

    unpack_path = '{}/files/kubernetes'.format(charm_dir)
    os.makedirs(unpack_path, exist_ok=True)
    cmd = ['tar', 'xfvz', archive, '-C', unpack_path]
    hookenv.log(cmd)
    check_call(cmd)

    apps = [
        {'name': 'kubelet', 'path': '/usr/local/bin'},
        {'name': 'kube-proxy', 'path': '/usr/local/bin'},
        {'name': 'kubectl', 'path': '/usr/local/bin'},
        {'name': 'loopback', 'path': '/opt/cni/bin'}
    ]

    for app in apps:
        unpacked = '{}/{}'.format(unpack_path, app['name'])
        app_path = os.path.join(app['path'], app['name'])
        install = ['install', '-v', '-D', unpacked, app_path]
        hookenv.log(install)
        check_call(install)

    set_state('kubernetes-worker.components.installed')


@when('kubernetes-worker.components.installed')
def set_app_version():
    ''' Declare the application version to juju '''
    cmd = ['kubelet', '--version']
    version = check_output(cmd)
    hookenv.application_version_set(version.split(b' v')[-1].rstrip())


@when('kubernetes-worker.components.installed')
@when_not('kube-dns.available')
def notify_user_transient_status():
    ''' Notify to the user we are in a transient state and the application
    is still converging. Potentially remotely, or we may be in a detached loop
    wait state '''

    # During deployment the worker has to start kubelet without cluster dns
    # configured. If this is the first unit online in a service pool waiting
    # to self host the dns pod, and configure itself to query the dns service
    # declared in the kube-system namespace

    hookenv.status_set('waiting', 'Waiting for cluster DNS.')


@when('kubernetes-worker.components.installed', 'kube-dns.available')
def charm_status(kube_dns):
    '''Update the status message with the current status of kubelet.'''
    update_kubelet_status()


def update_kubelet_status():
    ''' There are different states that the kubelt can be in, where we are
    waiting for dns, waiting for cluster turnup, or ready to serve
    applications.'''
    if (_systemctl_is_active('kubelet')):
        hookenv.status_set('active', 'Kubernetes worker running.')
    # if kubelet is not running, we're waiting on something else to converge
    elif (not _systemctl_is_active('kubelet')):
        hookenv.status_set('waiting', 'Waiting for kubelet to start.')


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


@when('kubernetes-worker.components.installed', 'kube-api-endpoint.available',
      'tls_client.ca.saved', 'tls_client.client.certificate.saved',
      'tls_client.client.key.saved', 'tls_client.server.certificate.saved',
      'tls_client.server.key.saved', 'kube-dns.available', 'cni.available')
def start_worker(kube_api, kube_dns, cni):
    ''' Start kubelet using the provided API and DNS info.'''
    servers = get_kube_api_servers(kube_api)
    # Note that the DNS server doesn't necessarily exist at this point. We know
    # what its IP will eventually be, though, so we can go ahead and configure
    # kubelet with that info. This ensures that early pods are configured with
    # the correct DNS even though the server isn't ready yet.

    dns = kube_dns.details()

    if (data_changed('kube-api-servers', servers) or
            data_changed('kube-dns', dns)):
        # Initialize a FlagManager object to add flags to unit data.
        opts = FlagManager('kubelet')
        # Append the DNS flags + data to the FlagManager object.

        opts.add('--cluster-dns', dns['sdn-ip'])  # FIXME sdn-ip needs a rename
        opts.add('--cluster-domain', dns['domain'])

        create_config(servers[0])
        render_init_scripts(servers)
        set_state('kubernetes-worker.config.created')
        restart_unit_services()
        update_kubelet_status()


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
                         '/etc/kubernetes/addons/default-http-backend.yaml')
        kubectl_manifest('delete',
                         '/etc/kubernetes/addons/ingress-replication-controller.yaml')  # noqa
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
            try:
                _apply_node_label(label, delete=True)
            except CalledProcessError:
                hookenv.log('Error removing node label {}'.format(label))
        # if the label is in user labels we do nothing here, it will get set
        # during the atomic update below.

    # Atomically set a label
    for label in user_labels:
        _apply_node_label(label)


def arch():
    '''Return the package architecture as a string. Raise an exception if the
    architecture is not supported by kubernetes.'''
    # Get the package architecture for this system.
    architecture = check_output(['dpkg', '--print-architecture']).rstrip()
    # Convert the binary result into a string.
    architecture = architecture.decode('utf-8')
    return architecture


def create_config(server):
    '''Create a kubernetes configuration for the worker unit.'''
    # Get the options from the tls-client layer.
    layer_options = layer.options('tls-client')
    # Get all the paths to the tls information required for kubeconfig.
    ca = layer_options.get('ca_certificate_path')
    key = layer_options.get('client_key_path')
    cert = layer_options.get('client_certificate_path')

    # Create kubernetes configuration in the default location for ubuntu.
    create_kubeconfig('/home/ubuntu/.kube/config', server, ca, key, cert,
                      user='ubuntu')
    # Make the config dir readable by the ubuntu users so juju scp works.
    cmd = ['chown', '-R', 'ubuntu:ubuntu', '/home/ubuntu/.kube']
    check_call(cmd)
    # Create kubernetes configuration in the default location for root.
    create_kubeconfig('/root/.kube/config', server, ca, key, cert,
                      user='root')
    # Create kubernetes configuration for kubelet, and kube-proxy services.
    create_kubeconfig(kubeconfig_path, server, ca, key, cert,
                      user='kubelet')


def render_init_scripts(api_servers):
    ''' We have related to either an api server or a load balancer connected
    to the apiserver. Render the config files and prepare for launch '''
    context = {}
    context.update(hookenv.config())

    layer_options = layer.options('tls-client')
    ca_cert_path = layer_options.get('ca_certificate_path')
    server_cert_path = layer_options.get('server_certificate_path')
    server_key_path = layer_options.get('server_key_path')

    unit_name = os.getenv('JUJU_UNIT_NAME').replace('/', '-')
    context.update({'kube_api_endpoint': ','.join(api_servers),
                    'JUJU_UNIT_NAME': unit_name})

    kubelet_opts = FlagManager('kubelet')
    kubelet_opts.add('--require-kubeconfig', None)
    kubelet_opts.add('--kubeconfig', kubeconfig_path)
    kubelet_opts.add('--network-plugin', 'cni')
    kubelet_opts.add('--anonymous-auth', 'false')
    kubelet_opts.add('--client-ca-file', ca_cert_path)
    kubelet_opts.add('--tls-cert-file', server_cert_path)
    kubelet_opts.add('--tls-private-key-file', server_key_path)
    context['kubelet_opts'] = kubelet_opts.to_s()

    kube_proxy_opts = FlagManager('kube-proxy')
    kube_proxy_opts.add('--kubeconfig', kubeconfig_path)
    context['kube_proxy_opts'] = kube_proxy_opts.to_s()

    os.makedirs('/var/lib/kubelet', exist_ok=True)

    render('kube-default', '/etc/default/kube-default', context)
    render('kubelet.defaults', '/etc/default/kubelet', context)
    render('kubelet.service', '/lib/systemd/system/kubelet.service', context)
    render('kube-proxy.defaults', '/etc/default/kube-proxy', context)
    render('kube-proxy.service', '/lib/systemd/system/kube-proxy.service',
           context)


def create_kubeconfig(kubeconfig, server, ca, key, certificate, user='ubuntu',
                      context='juju-context', cluster='juju-cluster'):
    '''Create a configuration for Kubernetes based on path using the supplied
    arguments for values of the Kubernetes server, CA, key, certificate, user
    context and cluster.'''
    # Create the config file with the address of the master server.
    cmd = 'kubectl config --kubeconfig={0} set-cluster {1} ' \
          '--server={2} --certificate-authority={3} --embed-certs=true'
    check_call(split(cmd.format(kubeconfig, cluster, server, ca)))
    # Create the credentials using the client flags.
    cmd = 'kubectl config --kubeconfig={0} set-credentials {1} ' \
          '--client-key={2} --client-certificate={3} --embed-certs=true'
    check_call(split(cmd.format(kubeconfig, user, key, certificate)))
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
    addon_path = '/etc/kubernetes/addons/{}'
    manifest = addon_path.format('default-http-backend.yaml')
    # Render the default http backend (404) replicationcontroller manifest
    render('default-http-backend.yaml', manifest, context)
    hookenv.log('Creating the default http backend.')
    kubectl_manifest('create', manifest)
    # Render the ingress replication controller manifest
    manifest = addon_path.format('ingress-replication-controller.yaml')
    render('ingress-replication-controller.yaml', manifest, context)
    if kubectl_manifest('create', manifest):
        hookenv.log('Creating the ingress replication controller.')
        set_state('kubernetes-worker.ingress.available')
        hookenv.open_port(80)
        hookenv.open_port(443)
    else:
        hookenv.log('Failed to create ingress controller. Will attempt again next update.')  # noqa
        hookenv.close_port(80)
        hookenv.close_port(443)


def restart_unit_services():
    '''Reload the systemd configuration and restart the services.'''
    # Tell systemd to reload configuration from disk for all daemons.
    call(['systemctl', 'daemon-reload'])
    # Ensure the services available after rebooting.
    call(['systemctl', 'enable', 'kubelet.service'])
    call(['systemctl', 'enable', 'kube-proxy.service'])
    # Restart the services.
    hookenv.log('Restarting kubelet, and kube-proxy.')
    call(['systemctl', 'restart', 'kubelet'])
    call(['systemctl', 'restart', 'kube-proxy'])


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
    command = ['kubectl', '--kubeconfig=' + kubeconfig_path] + list(args)
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
    services = ('kubelet', 'kube-proxy')

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
    services = ('kubelet', 'kube-proxy')

    # The current nrpe-external-master interface doesn't handle a lot of logic,
    # use the charm-helpers code for now.
    hostname = nrpe.get_nagios_hostname()
    nrpe_setup = nrpe.NRPE(hostname=hostname)

    for service in services:
        nrpe_setup.remove_check(shortname=service)


def _systemctl_is_active(application):
    ''' Poll systemctl to determine if the application is running '''
    cmd = ['systemctl', 'is-active', application]
    try:
        raw = check_output(cmd)
        return b'active' in raw
    except Exception:
        return False


def _apply_node_label(label, delete=False):
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
    check_call(split(cmd))


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
