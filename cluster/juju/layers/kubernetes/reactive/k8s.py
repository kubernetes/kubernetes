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

import os

from shlex import split
from shutil import copy2
from subprocess import check_call

from charms.docker.compose import Compose
from charms.reactive import hook
from charms.reactive import remove_state
from charms.reactive import set_state
from charms.reactive import when
from charms.reactive import when_not

from charmhelpers.core import hookenv
from charmhelpers.core.hookenv import is_leader
from charmhelpers.core.hookenv import status_set
from charmhelpers.core.templating import render
from charmhelpers.core import unitdata
from charmhelpers.core.host import chdir
from contextlib import contextmanager


@hook('config-changed')
def config_changed():
    '''If the configuration values change, remove the available states.'''
    config = hookenv.config()
    if any(config.changed(key) for key in config.keys()):
        hookenv.log('Configuration options have changed.')
        # Use the Compose class that encapsulates the docker-compose commands.
        compose = Compose('files/kubernetes')
        hookenv.log('Removing kubelet container and kubelet.available state.')
        # Stop and remove the Kubernetes kubelet container..
        compose.kill('kubelet')
        compose.rm('kubelet')
        # Remove the state so the code can react to restarting kubelet.
        remove_state('kubelet.available')
        hookenv.log('Removing proxy container and proxy.available state.')
        # Stop and remove the Kubernetes proxy container.
        compose.kill('proxy')
        compose.rm('proxy')
        # Remove the state so the code can react to restarting proxy.
        remove_state('proxy.available')

    if config.changed('version'):
        hookenv.log('Removing kubectl.downloaded state so the new version'
                    ' of kubectl will be downloaded.')
        remove_state('kubectl.downloaded')


@when('tls.server.certificate available')
@when_not('k8s.server.certificate available')
def server_cert():
    '''When the server certificate is available, get the server certificate from
    the charm unit data and write it to the proper directory. '''
    destination_directory = '/srv/kubernetes'
    # Save the server certificate from unitdata to /srv/kubernetes/server.crt
    save_certificate(destination_directory, 'server')
    # Copy the unitname.key to /srv/kubernetes/server.key
    copy_key(destination_directory, 'server')
    set_state('k8s.server.certificate available')


@when('tls.client.certificate available')
@when_not('k8s.client.certficate available')
def client_cert():
    '''When the client certificate is available, get the client certificate
    from the charm unitdata and write it to the proper directory. '''
    destination_directory = '/srv/kubernetes'
    if not os.path.isdir(destination_directory):
        os.makedirs(destination_directory)
        os.chmod(destination_directory, 0o770)
    # The client certificate is also available on charm unitdata.
    client_cert_path = 'easy-rsa/easyrsa3/pki/issued/client.crt'
    kube_cert_path = os.path.join(destination_directory, 'client.crt')
    if os.path.isfile(client_cert_path):
        # Copy the client.crt to /srv/kubernetes/client.crt
        copy2(client_cert_path, kube_cert_path)
    # The client key is only available on the leader.
    client_key_path = 'easy-rsa/easyrsa3/pki/private/client.key'
    kube_key_path = os.path.join(destination_directory, 'client.key')
    if os.path.isfile(client_key_path):
        # Copy the client.key to /srv/kubernetes/client.key
        copy2(client_key_path, kube_key_path)


@when('tls.certificate.authority available')
@when_not('k8s.certificate.authority available')
def ca():
    '''When the Certificate Authority is available, copy the CA from the
    /usr/local/share/ca-certificates/k8s.crt to the proper directory. '''
    # Ensure the /srv/kubernetes directory exists.
    directory = '/srv/kubernetes'
    if not os.path.isdir(directory):
        os.makedirs(directory)
        os.chmod(directory, 0o770)
    # Normally the CA is just on the leader, but the tls layer installs the
    # CA on all systems in the /usr/local/share/ca-certificates directory.
    ca_path = '/usr/local/share/ca-certificates/{0}.crt'.format(
              hookenv.service_name())
    # The CA should be copied to the destination directory and named 'ca.crt'.
    destination_ca_path = os.path.join(directory, 'ca.crt')
    if os.path.isfile(ca_path):
        copy2(ca_path, destination_ca_path)
        set_state('k8s.certificate.authority available')


@when('kubelet.available', 'proxy.available', 'cadvisor.available')
def final_messaging():
    '''Lower layers emit messages, and if we do not clear the status messaging
    queue, we are left with whatever the last method call sets status to. '''
    # It's good UX to have consistent messaging that the cluster is online
    if is_leader():
        status_set('active', 'Kubernetes leader running')
    else:
        status_set('active', 'Kubernetes follower running')


@when('kubelet.available', 'proxy.available', 'cadvisor.available')
@when_not('skydns.available')
def launch_skydns():
    '''Create a kubernetes service and resource controller for the skydns
    service. '''
    # Only launch and track this state on the leader.
    # Launching duplicate SkyDNS rc will raise an error
    if not is_leader():
        return
    cmd = "kubectl create -f files/manifests/skydns-rc.yml"
    check_call(split(cmd))
    cmd = "kubectl create -f files/manifests/skydns-svc.yml"
    check_call(split(cmd))
    set_state('skydns.available')


@when('docker.available')
@when_not('etcd.available')
def relation_message():
    '''Take over messaging to let the user know they are pending a relationship
    to the ETCD cluster before going any further. '''
    status_set('waiting', 'Waiting for relation to ETCD')


@when('etcd.available', 'tls.server.certificate available')
@when_not('kubelet.available', 'proxy.available')
def master(etcd):
    '''Install and run the hyperkube container that starts kubernetes-master.
    This actually runs the kubelet, which in turn runs a pod that contains the
    other master components. '''
    render_files(etcd)
    # Use the Compose class that encapsulates the docker-compose commands.
    compose = Compose('files/kubernetes')
    status_set('maintenance', 'Starting the Kubernetes kubelet container.')
    # Start the Kubernetes kubelet container using docker-compose.
    compose.up('kubelet')
    set_state('kubelet.available')
    # Open the secure port for api-server.
    hookenv.open_port(6443)
    status_set('maintenance', 'Starting the Kubernetes proxy container')
    # Start the Kubernetes proxy container using docker-compose.
    compose.up('proxy')
    set_state('proxy.available')
    status_set('active', 'Kubernetes started')


@when('proxy.available')
@when_not('kubectl.downloaded')
def download_kubectl():
    '''Download the kubectl binary to test and interact with the cluster.'''
    status_set('maintenance', 'Downloading the kubectl binary')
    version = hookenv.config()['version']
    cmd = 'wget -nv -O /usr/local/bin/kubectl https://storage.googleapis.com/' \
          'kubernetes-release/release/{0}/bin/linux/amd64/kubectl'
    cmd = cmd.format(version)
    hookenv.log('Downloading kubelet: {0}'.format(cmd))
    check_call(split(cmd))
    cmd = 'chmod +x /usr/local/bin/kubectl'
    check_call(split(cmd))
    set_state('kubectl.downloaded')
    status_set('active', 'Kubernetes installed')


@when('kubectl.downloaded')
@when_not('kubectl.package.created')
def package_kubectl():
    '''Package the kubectl binary and configuration to a tar file for users
    to consume and interact directly with Kubernetes.'''
    if not is_leader():
        return
    context = 'default-context'
    cluster_name = 'kubernetes'
    public_address = hookenv.unit_public_ip()
    directory = '/srv/kubernetes'
    key = 'client.key'
    ca = 'ca.crt'
    cert = 'client.crt'
    user = 'ubuntu'
    port = '6443'
    with chdir(directory):
        # Create the config file with the external address for this server.
        cmd = 'kubectl config set-cluster --kubeconfig={0}/config {1} ' \
              '--server=https://{2}:{3} --certificate-authority={4}'
        check_call(split(cmd.format(directory, cluster_name, public_address,
                                    port, ca)))
        # Create the credentials.
        cmd = 'kubectl config set-credentials --kubeconfig={0}/config {1} ' \
              '--client-key={2} --client-certificate={3}'
        check_call(split(cmd.format(directory, user, key, cert)))
        # Create a default context with the cluster.
        cmd = 'kubectl config set-context --kubeconfig={0}/config {1}' \
              ' --cluster={2} --user={3}'
        check_call(split(cmd.format(directory, context, cluster_name, user)))
        # Now make the config use this new context.
        cmd = 'kubectl config use-context --kubeconfig={0}/config {1}'
        check_call(split(cmd.format(directory, context)))
        # Copy the kubectl binary to this directory
        cmd = 'cp -v /usr/local/bin/kubectl {0}'.format(directory)
        check_call(split(cmd))

        # Create an archive with all the necessary files.
        cmd = 'tar -cvzf /home/ubuntu/kubectl_package.tar.gz ca.crt client.crt client.key config kubectl'  # noqa
        check_call(split(cmd))
        set_state('kubectl.package.created')


@when('proxy.available')
@when_not('cadvisor.available')
def start_cadvisor():
    '''Start the cAdvisor container that gives metrics about the other
    application containers on this system. '''
    compose = Compose('files/kubernetes')
    compose.up('cadvisor')
    set_state('cadvisor.available')
    status_set('active', 'cadvisor running on port 8088')
    hookenv.open_port(8088)


@when('sdn.available')
def gather_sdn_data():
    '''Get the Software Defined Network (SDN) information and return it as a
    dictionary.'''
    # SDN Providers pass data via the unitdata.kv module
    db = unitdata.kv()
    # Generate an IP address for the DNS provider
    subnet = db.get('sdn_subnet')
    if subnet:
        ip = subnet.split('/')[0]
        dns_server = '.'.join(ip.split('.')[0:-1]) + '.10'
        addedcontext = {}
        addedcontext['dns_server'] = dns_server
        return addedcontext
    return {}


def copy_key(directory, prefix):
    '''Copy the key from the easy-rsa/easyrsa3/pki/private directory to the
    specified directory. '''
    if not os.path.isdir(directory):
        os.makedirs(directory)
        os.chmod(directory, 0o770)
    # Must remove the path characters from the local unit name.
    path_name = hookenv.local_unit().replace('/', '_')
    # The key is not in unitdata it is in the local easy-rsa directory.
    local_key_path = 'easy-rsa/easyrsa3/pki/private/{0}.key'.format(path_name)
    key_name = '{0}.key'.format(prefix)
    # The key should be copied to this directory.
    destination_key_path = os.path.join(directory, key_name)
    # Copy the key file from the local directory to the destination.
    copy2(local_key_path, destination_key_path)


def render_files(reldata=None):
    '''Use jinja templating to render the docker-compose.yml and master.json
    file to contain the dynamic data for the configuration files.'''
    context = {}
    # Load the context manager with sdn and config data.
    context.update(gather_sdn_data())
    context.update(hookenv.config())
    if reldata:
        context.update({'connection_string': reldata.connection_string()})
    charm_dir = hookenv.charm_dir()
    rendered_kube_dir = os.path.join(charm_dir, 'files/kubernetes')
    if not os.path.exists(rendered_kube_dir):
        os.makedirs(rendered_kube_dir)
    rendered_manifest_dir = os.path.join(charm_dir, 'files/manifests')
    if not os.path.exists(rendered_manifest_dir):
        os.makedirs(rendered_manifest_dir)
    # Add the manifest directory so the docker-compose file can have.
    context.update({'manifest_directory': rendered_manifest_dir,
                    'private_address': hookenv.unit_get('private-address')})

    # Render the files/kubernetes/docker-compose.yml file that contains the
    # definition for kubelet and proxy.
    target = os.path.join(rendered_kube_dir, 'docker-compose.yml')
    render('docker-compose.yml', target, context)
    # Render the files/manifests/master.json that contains parameters for the
    # apiserver, controller, and controller-manager
    target = os.path.join(rendered_manifest_dir, 'master.json')
    render('master.json', target, context)
    # Render files/kubernetes/skydns-svc.yaml for SkyDNS service
    target = os.path.join(rendered_manifest_dir, 'skydns-svc.yml')
    render('skydns-svc.yml', target, context)
    # Render files/kubernetes/skydns-rc.yaml for SkyDNS pods
    target = os.path.join(rendered_manifest_dir, 'skydns-rc.yml')
    render('skydns-rc.yml', target, context)


def save_certificate(directory, prefix):
    '''Get the certificate from the charm unitdata, and write it to the proper
    directory. The parameters are: destination directory, and prefix to use
    for the key and certificate name.'''
    if not os.path.isdir(directory):
        os.makedirs(directory)
        os.chmod(directory, 0o770)
    # Grab the unitdata key value store.
    store = unitdata.kv()
    certificate_data = store.get('tls.{0}.certificate'.format(prefix))
    certificate_name = '{0}.crt'.format(prefix)
    # The certificate should be saved to this directory.
    certificate_path = os.path.join(directory, certificate_name)
    # write the server certificate out to the correct location
    with open(certificate_path, 'w') as fp:
        fp.write(certificate_data)
