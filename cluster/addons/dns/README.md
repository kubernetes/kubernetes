# SkyDNS Replication Controllers and Service templates

This directory contains the base templates that can be used to generate the
DNS templates in various other formats, including salt.

Due to a varied preference in templating language choices, the transform
Makefile in this directory should be enhanced to generate all required
formats from the base templates.

**NOTE WELL**: Developers, when you add a parameter you should also
update the various scripts that supply values for your new parameter.
Here is one way you might find those scripts:

```
cd kubernetes
find . -type f | xargs grep kubedns-rc.yaml
```

## Base Template files

These are the authoritative base templates.
Run `make' to generate the `salt` and `sed` yaml templates from these.

kubedns-rc.yaml.base
kubedns-svc.yaml.base

## Generated salt files

These files are appropriate for use in our `salt` based installations.

kubedns-rc.yaml.jinja
kubedns-svc.yaml.jinja

## Generated sed files

These files are appropriate for use where simple transformations are available,
such as running them through `sed`.

kubedns-rc.yaml.sed
kubedns-svc.yaml.sed

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/cluster/addons/dns/README.md?pixel)]()
