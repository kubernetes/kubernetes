# Experimental API

## What is the experimental API

The exerimental API is an API group indended for experimenting with and testing
APIs. To make this possible, unlike other API groups, there is no guarantee of
future support, compatibility, or even existence. We do this to lower the
development costs around making new APIs, to let experiments proceed without
ongoing maintenance costs.

## Where is the experimental API group?

It is hosted by kube-apiserver, at /apis/experimental/v1alpha1. `kubectl`
understands experimental objects, so they will work just like other objects in
the Kubernetes system. You'll have to explicitly say "experimental/" in front of
resource names, e.g. "experimental/daemonset". Because of the somewhat ephemeral
nature of experimental resources, we want users to explicitly type this to
create them.

## How do I enable the experimental API group?

A flag must be passed to the kube-apiserver binary. See the system
administration guide's [API group page](../admin-guide/api-groups.md).

## What might surprise me about objects I create in the experimental API group?

No developer effort will be spent keeping experimental objects working across
upgrades; that is the whole point of experimental APIs. Experimental objects
should be deleted before upgrading and recreated afterwards. For point-release
upgrades (e.g. 1.1 -> 1.2), this may be done for you automatically.
