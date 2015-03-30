#!/usr/bin/python

import subprocess
import re

def main():
    module = AnsibleModule(
        argument_spec = dict(
        ),
    )

    facts = {}

    result = {}
    result['rc'] = 0
    result['changed'] = False
    result['ansible_facts'] = facts

    args = ("rpm", "-q", "firewalld")
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    rc = popen.wait()
    facts['has_firewalld'] = False
    if rc == 0:
        facts['has_firewalld'] = True

    args = ("rpm", "-q", "iptables-services")
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    rc = popen.wait()
    facts['has_iptables'] = False
    if rc == 0:
        facts['has_iptables'] = True

    module.exit_json(**result)

# import module snippets
from ansible.module_utils.basic import *
main()
