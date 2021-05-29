{% panel style="success", title="Providing Feedback" %}
**Provide feedback at the [survey](https://www.surveymonkey.com/r/CLQBQHR)**
{% endpanel %}

{% panel style="info", title="TL;DR" %}
- [krew.sigs.k8s.io](https://krew.sigs.k8s.io/docs/user-guide/setup/install/) is a kubernetes sub-project to discover and manage plugins
{% endpanel %}

# Krew

By design, `kubectl` does not install plugins. This task is left to the kubernetes sub-project
[krew.sigs.k8s.io](https://krew.sigs.k8s.io/docs/user-guide/setup/install/) which needs to be installed separately.
Krew helps to

- discover plugins
- get updates for installed plugins
- remove plugins

## Installing krew

Krew should be used as a kubectl plugin. To set yourself up to using krew, you need to do two things:

1. Install git
1. Install krew as described on the project page [krew.sigs.k8s.io](https://krew.sigs.k8s.io/docs/user-guide/setup/install/).
1. Add the krew bin folder to your `PATH` environment variable. For example, in bash `export PATH="${KREW_ROOT:-$HOME/.krew}/bin:$PATH"`.

## Krew capabilities

{% method %}
Discover plugins
{% sample lang="yaml" %}
```bash
kubectl krew search
```
{% endmethod %}

{% method %}
Install a plugin
{% sample lang="yaml" %}
```bash
kubectl krew install access-matrix
```
{% endmethod %}

{% method %}
Upgrade all installed plugins
{% sample lang="yaml" %}
```bash
kubectl krew upgrade
```
{% endmethod %}

{% method %}
Show details about a plugin
{% sample lang="yaml" %}
```bash
kubectl krew info access-matrix
```
{% endmethod %}

{% method %}
Uninstall a plugin
{% sample lang="yaml" %}
```bash
kubectl krew uninstall access-matrix
```
{% endmethod %}
