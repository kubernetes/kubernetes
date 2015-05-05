#!/usr/bin/env python

# Copyright 2014 The Kubernetes Authors All rights reserved.
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

import re

import salt.exceptions
import salt.utils.ipaddr as ipaddr

def ensure(name, cidr, mtu=1460):
    '''
    Ensure that a bridge (named <name>) is configured for containers.

    Under the covers we will make sure that
      - The bridge exists
      - The MTU is set
      - The correct network is added to the bridge
      - iptables is set up for MASQUERADE for egress

    cidr:
        The cidr range in the form of 10.244.x.0/24
    mtu:
        The MTU to set on the interface
    '''
    ret = {'name': name, 'changes': {}, 'result': False, 'comment': ''}

    # This is a little hacky.  I should probably import a real library for this
    # but this'll work for now.
    try:
        cidr_network = ipaddr.IPNetwork(cidr, strict=True)
    except Exception:
        raise salt.exceptions.SaltInvocationError(
            'Invalid CIDR \'{0}\''.format(cidr))

    if cidr_network.version == 4:
        iptables_rule = {
            'table': 'nat',
            'chain': 'POSTROUTING',
            'rule': '-o eth0 -j MASQUERADE \! -d 10.0.0.0/8'
        }
    else:
        iptables_rule = None

    def bridge_exists(name):
        'Determine if a bridge exists already.'
        out = __salt__['cmd.run_stdout']('brctl show {0}'.format(name))
        for line in out.splitlines():
            # get rid of first line
            if line.startswith('bridge name'):
                continue
            # get rid of ^\n's
            vals = line.split()
            if not vals:
                continue
            if len(vals) > 1:
                return True
        return False

    def get_ip_addr_details(name):
        'For the given interface, get address details.'
        out = __salt__['cmd.run']('ip addr show dev {0}'.format(name))
        ret = { 'networks': [] }
        for line in out.splitlines():
            match = re.match(
                r'^\d*:\s+([\w.\-]+)(?:@)?([\w.\-]+)?:\s+<(.+)>.*mtu (\d+)',
                line)
            if match:
                iface, parent, attrs, mtu = match.groups()
                if 'UP' in attrs.split(','):
                    ret['up'] = True
                else:
                    ret['up'] = False
                if parent:
                    ret['parent'] = parent
                ret['mtu'] = int(mtu)
                continue
            cols = line.split()
            if len(cols) > 2 and cols[0] == 'inet':
                ret['networks'].append(cols[1])
        return ret


    def get_current_state():
        'Helper that returns a dict of current bridge state.'
        ret = {}
        ret['name'] = name
        ret['exists'] = bridge_exists(name)
        if ret['exists']:
            ret['details'] = get_ip_addr_details(name)
        else:
            ret['details'] = {}
        # This module function is strange and returns True if the rule exists.
        # If not, it returns a string with the error from the call to iptables.
        if iptables_rule:
            ret['iptables_rule_exists'] = \
              __salt__['iptables.check'](**iptables_rule) == True
        else:
            ret['iptables_rule_exists'] = True
        return ret

    desired_network = '{0}/{1}'.format(
        str(ipaddr.IPAddress(cidr_network._ip + 1)),
        str(cidr_network.prefixlen))

    current_state = get_current_state()

    if (current_state['exists']
        and current_state['details']['mtu'] == mtu
        and desired_network in current_state['details']['networks']
        and current_state['details']['up']
        and current_state['iptables_rule_exists']):
        ret['result'] = True
        ret['comment'] = 'System already in the correct state'
        return ret

    # The state of the system does need to be changed. Check if we're running
    # in ``test=true`` mode.
    if __opts__['test'] == True:
        ret['comment'] = 'The state of "{0}" will be changed.'.format(name)
        ret['changes'] = {
            'old': current_state,
            'new': 'Create and configure bridge'
        }

        # Return ``None`` when running with ``test=true``.
        ret['result'] = None

        return ret

    # Finally, make the actual change and return the result.
    if not current_state['exists']:
        __salt__['cmd.run']('brctl addbr {0}'.format(name))
    new_state = get_current_state()
    if new_state['details']['mtu'] != mtu:
        __salt__['cmd.run'](
            'ip link set dev {0} mtu {1}'.format(name, str(mtu)))
    new_state = get_current_state()
    if desired_network not in new_state['details']['networks']:
        __salt__['cmd.run'](
            'ip addr add {0} dev {1}'.format(desired_network, name))
    new_state = get_current_state()
    if not new_state['details']['up']:
        __salt__['cmd.run'](
            'ip link set dev {0} up'.format(name))
    new_state = get_current_state()
    if iptables_rule and not new_state['iptables_rule_exists']:
        __salt__['iptables.append'](**iptables_rule)
    new_state = get_current_state()

    ret['comment'] = 'The state of "{0}" was changed!'.format(name)

    ret['changes'] = {
        'old': current_state,
        'new': new_state,
    }

    ret['result'] = True

    return ret
