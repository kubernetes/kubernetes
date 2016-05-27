# SkyDNS Replication Controllers and Service templates

This directory contains the base UNDERSCORE templates that can be used
to generate the skydns-rc.yaml.in and skydns.rc.yaml.in needed in Salt format.

Due to a varied preference in templating language choices, the transform
Makefile in this directory should be enhanced to generate all required
formats from the base underscore templates.

## Base Template files

These are the authoritative base templates.
Run 'make' to generate the Salt and Sed yaml templates from these.

skydns-rc.yaml.base
skydns-svc.yaml.base

## Generated Salt files

skydns-rc.yaml.in
skydns-svc.yaml.in

## Generated Sed files

skydns-rc.yaml.sed
skydns-svc.yaml.sed

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/cluster/saltbase/salt/kube-dns/README.md?pixel)]()
