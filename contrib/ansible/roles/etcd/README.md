Role Name
=========

Configures an etcd cluster for an arbitrary number of hosts

Role Variables
--------------

TODO

Dependencies
------------

None

Example Playbook
----------------

    - hosts: etcd
      roles:
         - { etcd }

License
-------

MIT

Author Information
------------------

Scott Dodson <sdodson@redhat.com>, Tim St. Clair <tstclair@redhat.com>
Adapted from https://github.com/retr0h/ansible-etcd. We
should at some point submit a PR to merge this with that module.


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/contrib/ansible/roles/etcd/README.md?pixel)]()
