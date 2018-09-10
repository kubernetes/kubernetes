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
import re
import os
import subprocess
import hashlib
import json

from pathlib import Path
from subprocess import check_output, check_call
from socket import gethostname, getfqdn
from shlex import split
from subprocess import CalledProcessError
from charmhelpers.core import hookenv, unitdata
from charmhelpers.core import host
from charms.reactive import is_state, endpoint_from_flag
from time import sleep

db = unitdata.kv()
kubeclientconfig_path = '/root/.kube/config'
gcp_creds_env_key = 'GOOGLE_APPLICATION_CREDENTIALS'


def get_version(bin_name):
    """Get the version of an installed Kubernetes binary.

    :param str bin_name: Name of binary
    :return: 3-tuple version (maj, min, patch)

    Example::

        >>> `get_version('kubelet')
        (1, 6, 0)

    """
    cmd = '{} --version'.format(bin_name).split()
    version_string = subprocess.check_output(cmd).decode('utf-8')
    return tuple(int(q) for q in re.findall("[0-9]+", version_string)[:3])


def retry(times, delay_secs):
    """ Decorator for retrying a method call.

    Args:
        times: How many times should we retry before giving up
        delay_secs: Delay in secs

    Returns: A callable that would return the last call outcome
    """

    def retry_decorator(func):
        """ Decorator to wrap the function provided.

        Args:
            func: Provided function should return either True od False

        Returns: A callable that would return the last call outcome

        """
        def _wrapped(*args, **kwargs):
            res = func(*args, **kwargs)
            attempt = 0
            while not res and attempt < times:
                sleep(delay_secs)
                res = func(*args, **kwargs)
                if res:
                    break
                attempt += 1
            return res
        return _wrapped

    return retry_decorator


def calculate_resource_checksum(resource):
    ''' Calculate a checksum for a resource '''
    md5 = hashlib.md5()
    path = hookenv.resource_get(resource)
    if path:
        with open(path, 'rb') as f:
            data = f.read()
        md5.update(data)
    return md5.hexdigest()


def get_resource_checksum_db_key(checksum_prefix, resource):
    ''' Convert a resource name to a resource checksum database key. '''
    return checksum_prefix + resource


def migrate_resource_checksums(checksum_prefix, snap_resources):
    ''' Migrate resource checksums from the old schema to the new one '''
    for resource in snap_resources:
        new_key = get_resource_checksum_db_key(checksum_prefix, resource)
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


def check_resources_for_upgrade_needed(checksum_prefix, snap_resources):
    hookenv.status_set('maintenance', 'Checking resources')
    for resource in snap_resources:
        key = get_resource_checksum_db_key(checksum_prefix, resource)
        old_checksum = db.get(key)
        new_checksum = calculate_resource_checksum(resource)
        if new_checksum != old_checksum:
            return True
    return False


def calculate_and_store_resource_checksums(checksum_prefix, snap_resources):
    for resource in snap_resources:
        key = get_resource_checksum_db_key(checksum_prefix, resource)
        checksum = calculate_resource_checksum(resource)
        db.set(key, checksum)


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


def service_restart(service_name):
    hookenv.status_set('maintenance', 'Restarting {0} service'.format(
        service_name))
    host.service_restart(service_name)


def service_start(service_name):
    hookenv.log('Starting {0} service.'.format(service_name))
    host.service_stop(service_name)


def service_stop(service_name):
    hookenv.log('Stopping {0} service.'.format(service_name))
    host.service_stop(service_name)


def arch():
    '''Return the package architecture as a string. Raise an exception if the
    architecture is not supported by kubernetes.'''
    # Get the package architecture for this system.
    architecture = check_output(['dpkg', '--print-architecture']).rstrip()
    # Convert the binary result into a string.
    architecture = architecture.decode('utf-8')
    return architecture


def get_service_ip(service, namespace="kube-system", errors_fatal=True):
    cmd = "kubectl get service --namespace {} {} --output json".format(
        namespace, service)
    if errors_fatal:
        output = check_output(cmd, shell=True).decode()
    else:
        try:
            output = check_output(cmd, shell=True).decode()
        except CalledProcessError:
            return None
    svc = json.loads(output)
    return svc['spec']['clusterIP']


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


def create_kubeconfig(kubeconfig, server, ca, key=None, certificate=None,
                      user='ubuntu', context='juju-context',
                      cluster='juju-cluster', password=None, token=None,
                      keystone=False):
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
    if keystone:
        # create keystone user
        cmd = 'kubectl config --kubeconfig={0} ' \
              'set-credentials keystone_user'.format(kubeconfig)
        check_call(split(cmd))
        # manually add exec command until kubectl can do it for us
        with open(kubeconfig, "r") as f:
            content = f.read()
            content = content.replace("user: {}", """user:
      exec:
        command: "/snap/bin/client-keystone-auth"
        apiVersion: "client.authentication.k8s.io/v1alpha1"
""")
        with open(kubeconfig, "w") as f:
            f.write(content)
        # create keystone context
        cmd = 'kubectl config set-context --cluster={0} ' \
              '--user={1} juju-keystone'.format(kubeconfig, cluster)
        check_call(split(cmd))


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


def configure_kubernetes_service(key, service, base_args, extra_args_key):
    db = unitdata.kv()

    prev_args_key = key + service
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


def _snap_common_path(component):
    return Path('/var/snap/{}/common'.format(component))


def cloud_config_path(component):
    return _snap_common_path(component) / 'cloud-config.conf'


def _gcp_creds_path(component):
    return _snap_common_path(component) / 'gcp-creds.json'


def _daemon_env_path(component):
    return _snap_common_path(component) / 'environment'


def _cloud_endpoint_ca_path(component):
    return _snap_common_path(component) / 'cloud-endpoint-ca.crt'


def write_gcp_snap_config(component):
    # gcp requires additional credentials setup
    gcp = endpoint_from_flag('endpoint.gcp.ready')
    creds_path = _gcp_creds_path(component)
    with creds_path.open('w') as fp:
        os.fchmod(fp.fileno(), 0o600)
        fp.write(gcp.credentials)

    # create a cloud-config file that sets token-url to nil to make the
    # services use the creds env var instead of the metadata server, as
    # well as making the cluster multizone
    comp_cloud_config_path = cloud_config_path(component)
    comp_cloud_config_path.write_text('[Global]\n'
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


def write_openstack_snap_config(component):
    # openstack requires additional credentials setup
    openstack = endpoint_from_flag('endpoint.openstack.ready')

    lines = [
        '[Global]',
        'auth-url = {}'.format(openstack.auth_url),
        'region = {}'.format(openstack.region),
        'username = {}'.format(openstack.username),
        'password = {}'.format(openstack.password),
        'tenant-name = {}'.format(openstack.project_name),
        'domain-name = {}'.format(openstack.user_domain_name),
    ]
    if openstack.endpoint_tls_ca:
        cloud_endpoint_ca_path = _cloud_endpoint_ca_path(component)
        cloud_endpoint_ca_path.write_text(base64.b64decode(
            openstack.endpoint_tls_ca
        ).decode('utf-8'))
        lines.append('ca-file = {}'.format(str(cloud_endpoint_ca_path)))
    if any([openstack.subnet_id,
            openstack.floating_network_id,
            openstack.lb_method,
            openstack.manage_security_groups]):
        lines.append('')
        lines.append('[LoadBalancer]')
    if openstack.subnet_id:
        lines.append('subnet-id = {}'.format(openstack.subnet_id))
    if openstack.floating_network_id:
        lines.append('floating-network-id = {}'.format(
            openstack.floating_network_id))
    if openstack.lb_method:
        lines.append('lb-method = {}'.format(
            openstack.lb_method))
    if openstack.manage_security_groups:
        lines.append('manage-security-groups = {}'.format(
            openstack.manage_security_groups))

    comp_cloud_config_path = cloud_config_path(component)
    comp_cloud_config_path.write_text(''.join('{}\n'.format(l) for l in lines))


def write_azure_snap_config(component):
    azure = endpoint_from_flag('endpoint.azure.ready')
    comp_cloud_config_path = cloud_config_path(component)
    comp_cloud_config_path.write_text(json.dumps({
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
