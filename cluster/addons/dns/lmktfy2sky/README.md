# lmktfy2sky
==============

A bridge between LMKTFY and SkyDNS.  This will watch the lmktfy API for
changes in Services and then publish those changes to SkyDNS through etcd.

For now, this is expected to be run in a pod alongside the etcd and SkyDNS
containers.

## Namespaces

LMKTFY namespaces become another level of the DNS hierarchy.  See the
description of `-domain` below.

## Flags

`-domain`: Set the domain under which all DNS names will be hosted.  For
example, if this is set to `lmktfy.io`, then a service named "nifty" in the
"default" namespace would be exposed through DNS as
"nifty.default.lmktfy.io".

`-verbose`: Log additional information.

'-etcd_mutation_timeout': For how long the application will keep retrying etcd 
mutation (insertion or removal of a dns entry) before giving up and crashing.
