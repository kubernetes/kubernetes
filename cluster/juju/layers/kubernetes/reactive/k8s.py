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
from subprocess import call
from subprocess import check_call
from subprocess import check_output

from charms.docker.compose import Compose
from charms.reactive import hook
from charms.reactive import remove_state
from charms.reactive import set_state
from charms.reactive import when
from charms.reactive import when_any
from charms.reactive import when_not

from charmhelpers.core import hookenv
from charmhelpers.core.hookenv import is_leader
from charmhelpers.core.hookenv import leader_set
from charmhelpers.core.hookenv import leader_get
from charmhelpers.core.templating import render
from charmhelpers.core import unitdata
from charmhelpers.core.host import chdir

import tlslib


@when('leadership.is_leader')
def i_am_leader():
    '''The leader is the Kubernetes master node. '''
    leader_set({'master-address': hookenv.unit_private_ip()})


@when_not('tls.client.authorization.required')
def configure_easrsa():
    '''Require the tls layer to generate certificates with "clientAuth". '''
    # By default easyrsa generates the server certificates without clientAuth
    # Setting this state before easyrsa is configured ensures the tls layer is
    # configured to generate certificates with client authentication.
    set_state('tls.client.authorization.required')
    domain = hookenv.config().get('dns_domain')
    cidr = hookenv.config().get('cidr')
    sdn_ip = get_sdn_ip(cidr)
    # Create extra sans that the tls layer will add to the server cert.
    extra_sans = [
        sdn_ip,
        'kubernetes',
        'kubernetes.{0}'.format(domain),
        'kubernetes.default',
        'kubernetes.default.svc',
        'kubernetes.default.svc.{0}'.format(domain)
    ]
    unitdata.kv().set('extra_sans', extra_sans)


@hook('config-changed')
def config_changed():
    '''If the configuration values change, remove the available states.'''
    config = hookenv.config()
    if any(config.changed(key) for key in config.keys()):
        hookenv.log('The configuration options have changed.')
        # Use the Compose class that encapsulates the docker-compose commands.
        compose = Compose('files/kubernetes')
        if is_leader():
            hookenv.log('Removing master container and kubelet.available state.')  # noqa
            # Stop and remove the Kubernetes kubelet container.
            compose.kill('master')
            compose.rm('master')
            compose.kill('proxy')
            compose.rm('proxy')
            # Remove the state so the code can react to restarting kubelet.
            remove_state('kubelet.available')
        else:
            hookenv.log('Removing kubelet container and kubelet.available state.')  # noqa
            # Stop and remove the Kubernetes kubelet container.
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
        hookenv.log('The version changed removing the states so the new '
                    'version of kubectl will be downloaded.')
        remove_state('kubectl.downloaded')
        remove_state('kubeconfig.created')


@when('tls.server.certificate available')
@when_not('k8s.server.certificate available')
def server_cert():
    '''When the server certificate is available, get the server certificate
    from the charm unitdata and write it to the kubernetes directory. '''
    server_cert = '/srv/kubernetes/server.crt'
    server_key = '/srv/kubernetes/server.key'
    # Save the server certificate from unit data to the destination.
    tlslib.server_cert(None, server_cert, user='ubuntu', group='ubuntu')
    # Copy the server key from the default location to the destination.
    tlslib.server_key(None, server_key, user='ubuntu', group='ubuntu')
    set_state('k8s.server.certificate available')


@when('tls.client.certificate available')
@when_not('k8s.client.certficate available')
def client_cert():
    '''When the client certificate is available, get the client certificate
    from the charm unitdata and write it to the kubernetes directory. '''
    client_cert = '/srv/kubernetes/client.crt'
    client_key = '/srv/kubernetes/client.key'
    # Save the client certificate from the default location to the destination.
    tlslib.client_cert(None, client_cert, user='ubuntu', group='ubuntu')
    # Copy the client key from the default location to the destination.
    tlslib.client_key(None, client_key, user='ubuntu', group='ubuntu')
    set_state('k8s.client.certficate available')


@when('tls.certificate.authority available')
@when_not('k8s.certificate.authority available')
def ca():
    '''When the Certificate Authority is available, copy the CA from the
    default location to the /srv/kubernetes directory. '''
    ca_crt = '/srv/kubernetes/ca.crt'
    # Copy the Certificate Authority to the destination directory.
    tlslib.ca(None, ca_crt, user='ubuntu', group='ubuntu')
    set_state('k8s.certificate.authority available')


@when('kubelet.available', 'leadership.is_leader')
@when_not('kubedns.available', 'skydns.available')
def launch_dns():
    '''Create the "kube-system" namespace, the kubedns resource controller,
    and the kubedns service. '''
    hookenv.log('Creating kubernetes kubedns on the master node.')
    # Only launch and track this state on the leader.
    # Launching duplicate kubeDNS rc will raise an error
    # Run a command to check if the apiserver is responding.
    return_code = call(split('kubectl cluster-info'))
    if return_code != 0:
        hookenv.log('kubectl command failed, waiting for apiserver to start.')
        remove_state('kubedns.available')
        # Return without setting kubedns.available so this method will retry.
        return
    # Check for the "kube-system" namespace.
    return_code = call(split('kubectl get namespace kube-system'))
    if return_code != 0:
        # Create the kube-system namespace that is used by the kubedns files.
        check_call(split('kubectl create namespace kube-system'))
    # Check for the kubedns replication controller.
    return_code = call(split('kubectl get -f files/manifests/kubedns-rc.yaml'))
    if return_code != 0:
        # Create the kubedns replication controller from the rendered file.
        check_call(split('kubectl create -f files/manifests/kubedns-rc.yaml'))
    # Check for the kubedns service.
    return_code = call(split('kubectl get -f files/manifests/kubedns-svc.yaml'))
    if return_code != 0:
        # Create the kubedns service from the rendered file.
        check_call(split('kubectl create -f files/manifests/kubedns-svc.yaml'))
    set_state('kubedns.available')


@when('skydns.available', 'leadership.is_leader')
def convert_to_kubedns():
    '''Delete the skydns containers to make way for the kubedns containers.'''
    hookenv.log('Deleteing the old skydns deployment.')
    # Delete the skydns replication controller.
    return_code = call(split('kubectl delete rc kube-dns-v11'))
    # Delete the skydns service.
    return_code = call(split('kubectl delete svc kube-dns'))
    remove_state('skydns.available')


@when('docker.available')
@when_not('etcd.available')
def relation_message():
    '''Take over messaging to let the user know they are pending a relationship
    to the ETCD cluster before going any further. '''
    status_set('waiting', 'Waiting for relation to ETCD')


@when('kubeconfig.created')
@when('etcd.available')
@when_not('kubelet.available', 'proxy.available')
def start_kubelet(etcd):
    '''Run the hyperkube container that starts the kubernetes services.
    When the leader, run the master services (apiserver, controller, scheduler,
    proxy)
    using the master.json from the rendered manifest directory.
    When a follower, start the node services (kubelet, and proxy). '''
    render_files(etcd)
    # Use the Compose class that encapsulates the docker-compose commands.
    compose = Compose('files/kubernetes')
    status_set('maintenance', 'Starting the Kubernetes services.')
    if is_leader():
        compose.up('master')
        compose.up('proxy')
        set_state('kubelet.available')
        # Open the secure port for api-server.
        hookenv.open_port(6443)
    else:
        # Start the Kubernetes kubelet container using docker-compose.
        compose.up('kubelet')
        set_state('kubelet.available')
        # Start the Kubernetes proxy container using docker-compose.
        compose.up('proxy')
        set_state('proxy.available')
    status_set('active', 'Kubernetes services started')


@when('docker.available')
@when_not('kubectl.downloaded')
def download_kubectl():
    '''Download the kubectl binary to test and interact with the cluster.'''
    status_set('maintenance', 'Downloading the kubectl binary')
    version = hookenv.config()['version']
    cmd = 'wget -nv -O /usr/local/bin/kubectl https://storage.googleapis.com' \
          '/kubernetes-release/release/{0}/bin/linux/{1}/kubectl'
    cmd = cmd.format(version, arch())
    hookenv.log('Downloading kubelet: {0}'.format(cmd))
    check_call(split(cmd))
    cmd = 'chmod +x /usr/local/bin/kubectl'
    check_call(split(cmd))
    set_state('kubectl.downloaded')


@when('kubectl.downloaded', 'leadership.is_leader', 'k8s.certificate.authority available', 'k8s.client.certficate available')  # noqa
@when_not('kubeconfig.created')
def master_kubeconfig():
    '''Create the kubernetes configuration for the master unit. The master
    should create a package with the client credentials so the user can
    interact securely with the apiserver.'''
    hookenv.log('Creating Kubernetes configuration for master node.')
    directory = '/srv/kubernetes'
    ca = '/srv/kubernetes/ca.crt'
    key = '/srv/kubernetes/client.key'
    cert = '/srv/kubernetes/client.crt'
    # Get the public address of the apiserver so users can access the master.
    server = 'https://{0}:{1}'.format(hookenv.unit_public_ip(), '6443')
    # Create the client kubeconfig so users can access the master node.
    create_kubeconfig(directory, server, ca, key, cert)
    # Copy the kubectl binary to this directory.
    cmd = 'cp -v /usr/local/bin/kubectl {0}'.format(directory)
    check_call(split(cmd))
    # Use a context manager to run the tar command in a specific directory.
    with chdir(directory):
        # Create a package with kubectl and the files to use it externally.
        cmd = 'tar -cvzf /home/ubuntu/kubectl_package.tar.gz ca.crt ' \
              'client.key client.crt kubectl kubeconfig'
        check_call(split(cmd))

    # This sets up the client workspace consistently on the leader and nodes.
    node_kubeconfig()
    set_state('kubeconfig.created')


@when('kubectl.downloaded', 'k8s.certificate.authority available', 'k8s.server.certificate available')  # noqa
@when_not('kubeconfig.created', 'leadership.is_leader')
def node_kubeconfig():
    '''Create the kubernetes configuration (kubeconfig) for this unit.
    The the nodes will create a kubeconfig with the server credentials so
    the services can interact securely with the apiserver.'''
    hookenv.log('Creating Kubernetes configuration for worker node.')
    directory = '/var/lib/kubelet'
    ca = '/srv/kubernetes/ca.crt'
    cert = '/srv/kubernetes/server.crt'
    key = '/srv/kubernetes/server.key'
    # Get the private address of the apiserver for communication between units.
    server = 'https://{0}:{1}'.format(leader_get('master-address'), '6443')
    # Create the kubeconfig for the other services.
    kubeconfig = create_kubeconfig(directory, server, ca, key, cert)
    # Install the kubeconfig in the root user's home directory.
    install_kubeconfig(kubeconfig, '/root/.kube', 'root')
    # Install the kubeconfig in the ubunut user's home directory.
    install_kubeconfig(kubeconfig, '/home/ubuntu/.kube', 'ubuntu')
    set_state('kubeconfig.created')


@when('proxy.available')
@when_not('cadvisor.available')
def start_cadvisor():
    '''Start the cAdvisor container that gives metrics about the other
    application containers on this system. '''
    compose = Compose('files/kubernetes')
    compose.up('cadvisor')
    hookenv.open_port(8088)
    status_set('active', 'cadvisor running on port 8088')
    set_state('cadvisor.available')


@when('kubelet.available', 'kubeconfig.created')
@when_any('proxy.available', 'cadvisor.available', 'kubedns.available')
def final_message():
    '''Issue some final messages when the services are started. '''
    # TODO: Run a simple/quick health checks before issuing this message.
    status_set('active', 'Kubernetes running.')


def gather_sdn_data():
    '''Get the Software Defined Network (SDN) information and return it as a
    dictionary. '''
    sdn_data = {}
    # The dictionary named 'pillar' is a construct of the k8s template files.
    pillar = {}
    # SDN Providers pass data via the unitdata.kv module
    db = unitdata.kv()
    # Ideally the DNS address should come from the sdn cidr.
    subnet = db.get('sdn_subnet')
    if subnet:
        # Generate the DNS ip address on the SDN cidr (this is desired).
        pillar['dns_server'] = get_dns_ip(subnet)
    else:
        # There is no SDN cider fall back to the kubernetes config cidr option.
        pillar['dns_server'] = get_dns_ip(hookenv.config().get('cidr'))
    # The pillar['dns_domain'] value is used in the kubedns-rc.yaml
    pillar['dns_domain'] = hookenv.config().get('dns_domain')
    # Use a 'pillar' dictionary so we can reuse the upstream kubedns templates.
    sdn_data['pillar'] = pillar
    return sdn_data


def install_kubeconfig(kubeconfig, directory, user):
    '''Copy the a file from the target to a new directory creating directories
    if necessary. '''
    # The file and directory must be owned by the correct user.
    chown = 'chown {0}:{0} {1}'
    if not os.path.isdir(directory):
        os.makedirs(directory)
        # Change the ownership of the config file to the right user.
        check_call(split(chown.format(user, directory)))
    # kubectl looks for a file named "config" in the ~/.kube directory.
    config = os.path.join(directory, 'config')
    # Copy the kubeconfig file to the directory renaming it to "config".
    cmd = 'cp -v {0} {1}'.format(kubeconfig, config)
    check_call(split(cmd))
    # Change the ownership of the config file to the right user.
    check_call(split(chown.format(user, config)))


def create_kubeconfig(directory, server, ca, key, cert, user='ubuntu'):
    '''Create a configuration for kubernetes in a specific directory using
    the supplied arguments, return the path to the file.'''
    context = 'default-context'
    cluster_name = 'kubernetes'
    # Ensure the destination directory exists.
    if not os.path.isdir(directory):
        os.makedirs(directory)
    # The configuration file should be in this directory named kubeconfig.
    kubeconfig = os.path.join(directory, 'kubeconfig')
    # Create the config file with the address of the master server.
    cmd = 'kubectl config set-cluster --kubeconfig={0} {1} ' \
          '--server={2} --certificate-authority={3}'
    check_call(split(cmd.format(kubeconfig, cluster_name, server, ca)))
    # Create the credentials using the client flags.
    cmd = 'kubectl config set-credentials --kubeconfig={0} {1} ' \
          '--client-key={2} --client-certificate={3}'
    check_call(split(cmd.format(kubeconfig, user, key, cert)))
    # Create a default context with the cluster.
    cmd = 'kubectl config set-context --kubeconfig={0} {1} ' \
          '--cluster={2} --user={3}'
    check_call(split(cmd.format(kubeconfig, context, cluster_name, user)))
    # Make the config use this new context.
    cmd = 'kubectl config use-context --kubeconfig={0} {1}'
    check_call(split(cmd.format(kubeconfig, context)))

    hookenv.log('kubectl configuration created at {0}.'.format(kubeconfig))
    return kubeconfig


def get_dns_ip(cidr):
    '''Get an IP address for the DNS server on the provided cidr.'''
    # Remove the range from the cidr.
    ip = cidr.split('/')[0]
    # Take the last octet off the IP address and replace it with 10.
    return '.'.join(ip.split('.')[0:-1]) + '.10'


def get_sdn_ip(cidr):
    '''Get the IP address for the SDN gateway based on the provided cidr.'''
    # Remove the range from the cidr.
    ip = cidr.split('/')[0]
    # Remove the last octet and replace it with 1.
    return '.'.join(ip.split('.')[0:-1]) + '.1'


def render_files(reldata=None):
    '''Use jinja templating to render the docker-compose.yml and master.json
    file to contain the dynamic data for the configuration files.'''
    context = {}
    # Load the context data with SDN data.
    context.update(gather_sdn_data())
    # Add the charm configuration data to the context.
    context.update(hookenv.config())
    if reldata:
        connection_string = reldata.get_connection_string()
        # Define where the etcd tls files will be kept.
        etcd_dir = '/etc/ssl/etcd'
        # Create paths to the etcd client ca, key, and cert file locations.
        ca = os.path.join(etcd_dir, 'client-ca.pem')
        key = os.path.join(etcd_dir, 'client-key.pem')
        cert = os.path.join(etcd_dir, 'client-cert.pem')
        # Save the client credentials (in relation data) to the paths provided.
        reldata.save_client_credentials(key, cert, ca)
        # Update the context so the template has the etcd information.
        context.update({'etcd_dir': etcd_dir,
                        'connection_string': connection_string,
                        'etcd_ca': ca,
                        'etcd_key': key,
                        'etcd_cert': cert})

    charm_dir = hookenv.charm_dir()
    rendered_kube_dir = os.path.join(charm_dir, 'files/kubernetes')
    if not os.path.exists(rendered_kube_dir):
        os.makedirs(rendered_kube_dir)
    rendered_manifest_dir = os.path.join(charm_dir, 'files/manifests')
    if not os.path.exists(rendered_manifest_dir):
        os.makedirs(rendered_manifest_dir)

    # Update the context with extra values, arch, manifest dir, and private IP.
    context.update({'arch': arch(),
                    'master_address': leader_get('master-address'),
                    'manifest_directory': rendered_manifest_dir,
                    'public_address': hookenv.unit_get('public-address'),
                    'private_address': hookenv.unit_get('private-address')})

    # Adapted from: http://kubernetes.io/docs/getting-started-guides/docker/
    target = os.path.join(rendered_kube_dir, 'docker-compose.yml')
    # Render the files/kubernetes/docker-compose.yml file that contains the
    # definition for kubelet and proxy.
    render('docker-compose.yml', target, context)

    if is_leader():
        # Source: https://github.com/kubernetes/...master/cluster/images/hyperkube  # noqa
        target = os.path.join(rendered_manifest_dir, 'master.json')
        # Render the files/manifests/master.json that contains parameters for
        # the apiserver, controller, and controller-manager
        render('master.json', target, context)
        # Source: ...cluster/addons/dns/skydns-svc.yaml.in
        target = os.path.join(rendered_manifest_dir, 'kubedns-svc.yaml')
        # Render files/kubernetes/kubedns-svc.yaml for the DNS service.
        render('kubedns-svc.yaml', target, context)
        # Source: ...cluster/addons/dns/skydns-rc.yaml.in
        target = os.path.join(rendered_manifest_dir, 'kubedns-rc.yaml')
        # Render files/kubernetes/kubedns-rc.yaml for the DNS pod.
        render('kubedns-rc.yaml', target, context)


def status_set(level, message):
    '''Output status message with leadership information.'''
    if is_leader():
        message = '{0} (master) '.format(message)
    hookenv.status_set(level, message)


def arch():
    '''Return the package architecture as a string. Raise an exception if the
    architecture is not supported by kubernetes.'''
    # Get the package architecture for this system.
    architecture = check_output(['dpkg', '--print-architecture']).rstrip()
    # Convert the binary result into a string.
    architecture = architecture.decode('utf-8')
    # Validate the architecture is supported by kubernetes.
    if architecture not in ['amd64', 'arm', 'arm64', 'ppc64le']:
        message = 'Unsupported machine architecture: {0}'.format(architecture)
        status_set('blocked', message)
        raise Exception(message)
    return architecture
