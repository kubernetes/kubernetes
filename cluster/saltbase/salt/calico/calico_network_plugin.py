#!/bin/python
from collections import namedtuple
import json
import os
import sys
from subprocess import check_output, CalledProcessError
import requests
import sh

# Append to existing env, to avoid losing PATH etc.
# TODO-PAT: This shouldn't be hardcoded
env = os.environ.copy()
env['ETCD_AUTHORITY'] = 'kubernetes-master:6666'
calicoctl = sh.Command('/home/vagrant/calicoctl').bake(_env=env)

ETCD_AUTHORITY_ENV = "ETCD_AUTHORITY"
PROFILE_LABEL = 'CALICO_PROFILE'
ETCD_PROFILE_PATH = '/calico/'
AllowRule = namedtuple('AllowRule', ['port', 'proto', 'source'])


class NetworkPlugin():
    def __init__(self):
        self.pod_name = None
        self.docker_id = None

    def create(self, args):
        """"Create a pod."""
        # Calicoctl only
        self.pod_name = args[3].replace('-', '_')
        self.docker_id = args[4]

        print('Configuring docker container %s' % self.docker_id)

        try:
            self._configure_interface()
            self._configure_profile()
        except CalledProcessError as e:
            print('Error code %d creating pod networking: %s\n%s' % (
                e.returncode, e.output, e))
            sys.exit(1)

    def delete(self, args):
        """Cleanup after a pod."""
        self.docker_id = args[4]

        # Remove the profile for the workload.
        calicoctl('container', 'remove', self.docker_id)

    def _configure_interface(self):
        """Configure the Calico interface for a pod."""
        ip = self._read_docker_ip()
        self._delete_docker_interface()
        print('Configuring Calico networking.')
        print(calicoctl('container', 'add', self.docker_id, ip))
        print('Finished configuring network interface')

    def _read_docker_ip(self):
        """Get the ID for the pod's infra container."""
        ip = check_output([
            'docker', 'inspect', '-format', '{{ .NetworkSettings.IPAddress }}',
            self.docker_id
        ])
        # Clean trailing whitespace (expect a '\n' at least).
        ip = ip.strip()

        print('Docker-assigned IP was %s' % ip)
        return ip

    def _delete_docker_interface(self):
        """Delete the existing veth connecting to the docker bridge."""
        print('Deleting eth0')

        # Get the PID of the container.
        pid = check_output([
            'docker', 'inspect', '-format', '{{ .State.Pid }}',
            self.docker_id
        ])
        # Clean trailing whitespace (expect a '\n' at least).
        pid = pid.strip()

        # Set up a link to the container's netns.
        print(check_output(['mkdir', '-p', '/var/run/netns']))
        netns_file = '/var/run/netns/' + pid
        if not os.path.isfile(netns_file):
            print(check_output(['ln', '-s', '/proc/' + pid + '/ns/net',
                                netns_file]))

        # Reach into the netns and delete the docker-allocated interface.
        print(check_output(['ip', 'netns', 'exec', pid,
                            'ip', 'link', 'del', 'eth0']))

        # Clean up after ourselves (don't want to leak netns files)
        print(check_output(['rm', netns_file]))

    def _configure_profile(self):
        """
        Configure the calico profile for a pod.

        Currently assumes one pod with each name.
        """
        calicoctl('profile', 'add', self.pod_name)
        pod = self._get_pod_config()
        profile_name = self.pod_name

        self._apply_rules(profile_name, pod)

        self._apply_tags(profile_name, pod)

        # Also add the workload to the profile.
        calicoctl('profile', profile_name, 'member', 'add', self.docker_id)
        print('Finished configuring profile.')

    def _get_pod_ports(self, pod):
        """
        Get the list of ports on containers in the Pod.

        :return list ports: the Kubernetes ContainerPort objects for the pod.
        """
        ports = []
        for container in pod['spec']['containers']:
            try:
                more_ports = container['ports']
                print('Adding ports %s' % more_ports)
                ports.extend(more_ports)
            except KeyError:
                pass
        return ports

    def _get_pod_config(self):
        pods = self._get_pods()

        for pod in pods:
            print('Processing pod %s' % pod)
            if pod['metadata']['name'].replace('-', '_') == self.pod_name:
                this_pod = pod
                break
        else:
            raise KeyError('Pod not found: ' + self.pod_name)
        print('Got pod data %s' % this_pod)
        return this_pod

    def _get_pods(self):
        """
        Get the list of pods from the Kube API server.

        :return list pods: A list of Pod JSON API objects
        """
        bearer_token = self._get_api_token()
        session = requests.Session()
        session.headers.update({'Authorization': 'Bearer ' + bearer_token})
        response = session.get(
            'https://kubernetes-master:6443/api/v1beta3/pods',
            verify=False,
        )
        response_body = response.text
        # The response body contains some metadata, and the pods themselves
        # under the 'items' key.
        pods = json.loads(response_body)['items']
        print('Got pods %s' % pods)
        return pods

    def _get_api_token(self):
        """
        Get the kubelet Bearer token for this node, used for HTTPS auth.
        :return: The token.
        :rtype: str
        """
        with open('/var/lib/kubelet/kubernetes_auth') as f:
            json_string = f.read()
        print('Got kubernetes_auth: ' + json_string)

        auth_data = json.loads(json_string)
        return auth_data['BearerToken']

    def _generate_rules(self):
        """
        Generate the Profile rules that have been specified on the Pod's ports.

        We only create a Rule for a port if it has 'allowFrom' specified.

        The Rule is structured to match the Calico etcd format.

        :param profile_name: The name of the Profile being generated
        :type profile_name: string
        :param ports: a list of ContainerPort objecs.
        :type ports: list
        :return list() rules: the rules to be added to the Profile.
        """
        inbound_rules = [
            {
                'action': 'allow',
            },
        ]

        outbound_rules = [
            {
                'action': 'allow',
            },
        ]
        return inbound_rules, outbound_rules

    def _generate_profile_json(self, profile_name, rules):
        """
        Given a list of of Calico rules, generate a Calico Profile JSON blob
        implementing those rules.

        :param profile_name: The name of the Calico profile
        :type profile_name: string
        :param rules: A tuple of (inbound, outbound) Calico rules
        :type rules: tuple
        :return: A JSON blob ready to be loaded by calicoctl
        :rtype: str
        """
        inbound, outbound = rules
        profile = {
            'id': profile_name,
            'inbound_rules': inbound,
            'outbound_rules': outbound,
        }
        profile_json = json.dumps(profile, indent=2)
        print('Final profile "%s": %s' % (profile_name, profile_json))
        return profile_json

    def _apply_rules(self, profile_name, pod):
        """
        Generate a new profile with the specified rules.

        This contains inbound allow rules for all the ports we gathered,
        plus a default 'allow from <profile_name>' to allow traffic within a
        profile group.

        :param profile_name: The profile to update
        :type profile_name: string
        :param pod: The config dictionary for the pod being created.
        :type pod: dict
        :return:
        """
        rules = self._generate_rules()
        profile_json = self._generate_profile_json(profile_name, rules)

        # Pipe the Profile JSON into the calicoctl command to update the rule.
        calicoctl('profile', profile_name, 'rule', 'update',
                  _in=profile_json)
        print('Finished applying rules.')

    def _apply_tags(self, profile_name, pod):
        """
        Extract the label KV pairs from the pod config, and apply each as a
        tag in the pod's profile.

        :param profile_name: The name of the Calico profile.
        :type profile_name: string
        :param pod: The config dictionary for the pod being created.
        :type pod: dict
        :return:
        """
        try:
            labels = pod['metadata']['labels']
        except KeyError:
            # If there are no labels, there's no more work to do.
            print('No labels found in pod %s' % pod)
            return

        for k, v in labels.iteritems():
            tag = '%s_%s' % (k, v)
            tag = tag.replace('-', '_')
            print('Adding tag ' + tag)
            try:
                calicoctl('profile', profile_name, 'tag', 'add', tag)
            except sh.ErrorReturnCode as e:
                print('Could not create tag %s.\n%s' % (tag, e))
        print('Finished applying tags.')

if __name__ == '__main__':
    print('Args: %s' % sys.argv)
    mode = sys.argv[1]


    if mode == 'init':
        print('No initialization work to perform')
    elif mode == 'setup':
        print('Executing Calico pod-creation hook')
        NetworkPlugin().create(sys.argv)
    elif mode == 'teardown':
        print('Executing Calico pod-deletion hook')
        NetworkPlugin().delete(sys.argv)
