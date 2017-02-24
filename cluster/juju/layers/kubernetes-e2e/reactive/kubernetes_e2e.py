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

from charms import layer

from charms.reactive import hook
from charms.reactive import is_state
from charms.reactive import remove_state
from charms.reactive import set_state
from charms.reactive import when
from charms.reactive import when_not

from charmhelpers.core import hookenv

from shlex import split

from subprocess import call
from subprocess import check_call
from subprocess import check_output


@hook('upgrade-charm')
def reset_delivery_states():
    ''' Remove the state set when resources are unpacked. '''
    remove_state('kubernetes-e2e.installed')


@when('kubernetes-e2e.installed')
def messaging():
    ''' Probe our relations to determine the propper messaging to the
    end user '''

    missing_services = []
    if not is_state('kubernetes-master.available'):
        missing_services.append('kubernetes-master')
    if not is_state('certificates.available'):
        missing_services.append('certificates')

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


@when_not('kubernetes-e2e.installed')
def install_kubernetes_e2e():
    ''' Deliver the e2e and kubectl components from the binary resource stream
    packages declared in the charm '''
    charm_dir = os.getenv('CHARM_DIR')
    arch = determine_arch()

    # Get the resource via resource_get
    resource = 'e2e_{}'.format(arch)
    try:
        archive = hookenv.resource_get(resource)
    except Exception:
        message = 'Error fetching the {} resource.'.format(resource)
        hookenv.log(message)
        hookenv.status_set('blocked', message)
        return

    if not archive:
        hookenv.log('Missing {} resource.'.format(resource))
        hookenv.status_set('blocked', 'Missing {} resource.'.format(resource))
        return

    # Handle null resource publication, we check if filesize < 1mb
    filesize = os.stat(archive).st_size
    if filesize < 1000000:
        hookenv.status_set('blocked',
                           'Incomplete {} resource.'.format(resource))
        return

    hookenv.status_set('maintenance',
                       'Unpacking {} resource.'.format(resource))

    unpack_path = '{}/files/kubernetes'.format(charm_dir)
    os.makedirs(unpack_path, exist_ok=True)
    cmd = ['tar', 'xfvz', archive, '-C', unpack_path]
    hookenv.log(cmd)
    check_call(cmd)

    services = ['e2e.test', 'ginkgo', 'kubectl']

    for service in services:
        unpacked = '{}/{}'.format(unpack_path, service)
        app_path = '/usr/local/bin/{}'.format(service)
        install = ['install', '-v', unpacked, app_path]
        call(install)

    set_state('kubernetes-e2e.installed')


@when('tls_client.ca.saved', 'tls_client.client.certificate.saved',
      'tls_client.client.key.saved', 'kubernetes-master.available',
      'kubernetes-e2e.installed')
@when_not('kubeconfig.ready')
def prepare_kubeconfig_certificates(master):
    ''' Prepare the data to feed to create the kubeconfig file. '''

    layer_options = layer.options('tls-client')
    # Get all the paths to the tls information required for kubeconfig.
    ca = layer_options.get('ca_certificate_path')
    key = layer_options.get('client_key_path')
    cert = layer_options.get('client_certificate_path')

    servers = get_kube_api_servers(master)

    # pedantry
    kubeconfig_path = '/home/ubuntu/.kube/config'

    # Create kubernetes configuration in the default location for ubuntu.
    create_kubeconfig('/root/.kube/config', servers[0], ca, key, cert,
                      user='root')
    create_kubeconfig(kubeconfig_path, servers[0], ca, key, cert,
                      user='ubuntu')
    # Set permissions on the ubuntu users kubeconfig to ensure a consistent UX
    cmd = ['chown', 'ubuntu:ubuntu', kubeconfig_path]
    check_call(cmd)

    set_state('kubeconfig.ready')


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
