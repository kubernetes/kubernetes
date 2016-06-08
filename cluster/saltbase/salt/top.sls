base:
  '*':
    - base
    - debian-auto-upgrades
    - salt-helpers
{% if grains.get('cloud') == 'aws' %}
    - ntp
{% endif %}
{% if pillar.get('e2e_storage_test_environment', '').lower() == 'true' %}
    - e2e
{% endif %}

  'roles:kubernetes-pool':
    - match: grain
    - pool

  'roles:kubernetes-master':
    - match: grain
    - master
