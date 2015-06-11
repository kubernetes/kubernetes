# Glossary of terms

##### Bootstrap
Initialize a Juju Environment so that Services may be deployed.
Bootstrapping an environment will provision a new machine in the environment
and run the Juju state server on that machine also called the bootstrap node.

##### Bundle
A set of Charms, configuration, and corresponding Relations that can be
deployed together as a single step.  Bundles are defined in the YAML format.

##### Charm
Charms are the basic building components of Juju. A charm provides the
definition of the Service, including its metadata, dependencies to other
Services, as well as the logic for management of the Service lifecycle. A
charm encapsulates the configuration management, how it interfaces
with other Services, lifecycle management, and peering/scaling up and down.

##### Environment
An Environment is a configured location where Service can be deployed. The  environment configuration is defined in the `~/.juju/environments.yaml`
file.

##### Interface
Interfaces are loosely typed contracts between Services that establish an
agreement of what is going to be sent across the wire between Services to
establish a relationship.

##### Juju
Juju is an open source universal model for Service oriented orchestration tool
developed by Canonical. Juju allows you to deploy, configure, manage, maintain
and scale cloud Services quickly on public clouds as well as on physical
servers.

##### Relation
Relations are the way in which Juju Services communicate with other Services,
and the way the topology of Services is connected.  A Charm defines what
relations it `provides` and what relations that it can connect with `requires`
Relations are a bi-directional communication pipeline, written declaratively in
charms in a 'provides' and 'requires' stanza. These relational interfaces
dictate how Service s interact with one another.

##### Repository
A location where multiple Charms are stored. Repositories may be as simple as
a directory structure on a local disk, or as complex as a rich smart server
supporting remote searching and so on.

##### Service
A Service is any application (or set of applications) that is integrated into
the framework as an individual component which should generally be joined with
other components to perform a more complex goal.
