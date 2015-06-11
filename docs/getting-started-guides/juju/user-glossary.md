---
layout: default
title: Glossary
category: User Docs
permalink: /user/glossary.html
---

##### Juju
Juju is a service orchestration tool developed and sponsored by Canonical. It
interfaces with with existing configuration management solutions in a language
agnostic way, and drives system deployments with a declarative, event driven
model.


##### Charm
Charms are the basic building components of Juju. They encapsulate the concerns
of a service. By concerns we mean: configuration management, how it interfaces
with other services, lifecycle management, and peering/scaling up and down.


##### Bundle
A bundle is a set of charms and relations with in Juju.  A bundle is a 
specific configuration corresponding relations that can be deployed together 
in one step.  Bundles are defined in text files in YAML format. 


##### Relation
Relations are a bi-directional communication pipeline, written declaratively in
charms in a 'provides' and 'requires' stanza. These relational interfaces
dictate how services interact with one another.

##### Interface

Interfaces are loosely typed contracts between services that establish an
agreement of what is going to be sent across the wire between services to
establish a relationship.

