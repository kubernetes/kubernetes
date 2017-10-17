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

from charms import layer
from charms.layer import snap

from charms.reactive import hook
from charms.reactive import is_state
from charms.reactive import set_state
from charms.reactive import when
from charms.reactive import when_not
from charms.reactive.helpers import data_changed

from charmhelpers.core import hookenv

from shlex import split

from subprocess import check_call
from subprocess import check_output


USER = 'system:e2e'


@hook('upgrade-charm')
def reset_delivery_states():
    ''' Remove the state set when resources are unpacked. '''
    install_snaps()


@when('kubernetes-e2e.installed')
def report_status():
    ''' Report the status of the charm. '''
    messaging()


def messaging():
    ''' Probe our relations to determine the propper messaging to the
    end user '''

    missing_services = []
    if not is_state('kubernetes-master.available'):
        missing_services.append('kubernetes-master:http')
    if not is_state('certificates.available'):
        missing_services.append('certificates')
    if not is_state('kubeconfig.ready'):
        missing_services.append('kubernetes-master:kube-control')

    if missing_services:
        if len(missing_services) > 1:
            subject = 'relations'
        else:
            subject = 'relation'

        services = ','.join(missing_services)
        message = 'Missing {0}: {1}'.format(subject, services)
        hookenv.status_set('blocked', message)
        return

    hookenv.status_set('active', 'Ready to test.')


@when('config.changed.channel')
def channel_changed():
    install_snaps()


def install_snaps():
    ''' Deliver the e2e and kubectl components from the binary resource stream
    packages declared in the charm '''
    channel = hookenv.config('channel')
    hookenv.status_set('maintenance', 'Installing kubectl snap')
    snap.install('kubectl', channel=channel, classic=True)
    hookenv.status_set('maintenance', 'Installing kubernetes-test snap')
    snap.install('kubernetes-test', channel=channel, classic=True)
    set_state('kubernetes-e2e.installed')


@when('tls_client.ca.saved', 'tls_client.client.certificate.saved',
      'tls_client.client.key.saved', 'kubernetes-master.available',
      'kubernetes-e2e.installed', 'e2e.auth.bootstrapped',
      'kube-control.auth.available')
@when_not('kubeconfig.ready')
def prepare_kubeconfig_certificates(master, kube_control):
    ''' Prepare the data to feed to create the kubeconfig file. '''

    layer_options = layer.options('tls-client')
    # Get all the paths to the tls information required for kubeconfig.
    ca = layer_options.get('ca_certificate_path')
    creds = kube_control.get_auth_credentials(USER)
    data_changed('kube-control.creds', creds)

    servers = get_kube_api_servers(master)

    # pedantry
    kubeconfig_path = '/home/ubuntu/.kube/config'

    # Create kubernetes configuration in the default location for ubuntu.
    create_kubeconfig('/root/.kube/config', servers[0], ca,
                      token=creds['client_token'], user='root')
    create_kubeconfig(kubeconfig_path, servers[0], ca,
                      token=creds['client_token'], user='ubuntu')
    # Set permissions on the ubuntu users kubeconfig to ensure a consistent UX
    cmd = ['chown', 'ubuntu:ubuntu', kubeconfig_path]
    check_call(cmd)
    messaging()
    set_state('kubeconfig.ready')


@when('kube-control.connected')
def request_credentials(kube_control):
    """ Request authorization creds."""

    # Ask for a user, although we will be using the 'client_token'
    kube_control.set_auth_request(USER)


@when('kube-control.auth.available')
def catch_change_in_creds(kube_control):
    """Request a service restart in case credential updates were detected."""
    creds = kube_control.get_auth_credentials(USER)
    if creds \
            and data_changed('kube-control.creds', creds) \
            and creds['user'] == USER:
        set_state('e2e.auth.bootstrapped')


@when('kubernetes-e2e.installed', 'kubeconfig.ready')
def set_app_version():
    ''' Declare the application version to juju '''
    cmd = ['kubectl', 'version', '--client']
    from subprocess import CalledProcessError
    try:
        version = check_output(cmd).decode('utf-8')
    except CalledProcessError:
        message = "Missing kubeconfig causes errors. Skipping version set."
        hookenv.log(message)
        return
    git_version = version.split('GitVersion:"v')[-1]
    version_from = git_version.split('",')[0]
    hookenv.application_version_set(version_from.rstrip())


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


def get_kube_api_servers(master):
    '''Return the kubernetes api server address and port for this
    relationship.'''
    hosts = []
    # Iterate over every service from the relation object.
    for service in master.services():
        for unit in service['hosts']:
            hosts.append('https://{0}:{1}'.format(unit['hostname'],
                                                  unit['port']))
    return hosts


def determine_arch():
    ''' dpkg wrapper to surface the architecture we are tied to'''
    cmd = ['dpkg', '--print-architecture']
    output = check_output(cmd).decode('utf-8')

    return output.rstrip()
