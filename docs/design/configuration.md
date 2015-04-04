# Kubernetes Component Configuration

## Abstract

A generic ``Configuration`` object to store command-line flags that is used for
configuration of ``Kubernetes`` components living on top of ``etcd`` is proposed
in this proposal.

The main focus of this proposal to solve dynamic configuration problem of components
and encapsulate configuration information to increase overall system quality and to
create a flexible configuration model for ``Kubernetes`` components.

``Kubernetes`` component configuration happens on the command line, and this proposal
builds on that fundamental principle to be coherent and easy for admins to comprehend.

## Problem

Currently command-line flags are used for component configurations in ``Kubernetes``.
But, this approach is limiting in a distributed computing environment. Specifically:

* Hard to synchronize across replicated components.
* Hard to change dynamically as needed by continuous services.
* Hard to change in different versions that may support different configuration options.

## Solution

A new ``Configuration`` resource is proposed to solve the aforementioned problems.
It provides flags to the ``Kubernetes`` components, which supplement the flags
specified on their command lines.

This resource is stored in ``etcd`` and accessible to interested/necessary components
through the API server. Components look up ``Configuration`` when they start and
optionally watch for changes to that configuration.

``Configuration`` resources are identified by their ``TypeMeta`` ``name``. By default
each ``Kubernetes`` component uses configuration associated with its own well-known
executable name, (eg: ```scheduler```). This can be overridden for some components (eg:
various kubelet's may use different configurations if desired).

Some per-component configuration is always necessary, such as ```--kubeconfig=xxx```
which describes how to connect to the API server. Once a component has contacted
the API server it supplements its command line flags from the Configuration stored in
API server.

Initially after implementing this proposal most configuration flags will be static,
that is changes to configuration will only affect components when they start up.
However over time, and as follow up work, ``Kubernetes`` components will choose to
watch certain flags for changes and react appropriately.

### Advantages

* Reusable across different components.
* Easy distribution through the API server.
* By leveraging the power of ``/watch API`` along with the new resource, we can change component configurations dynamically.
* Layer of abstraction that gets rid of the global state currently used for command-line flags.

### API Resource

A new resource for ``Configuration`` will be added to the ``API``:

```go
        type ConfigurationFlag struct {
                Flag string `json:"flag"`
                Value string `json:"value,omitempty"` // Optional, empty string if not present
        }

	type Configuration struct {
		TypeMeta   `json:",inline"`
		ObjectMeta `json:"metadata,omitempty"`

		// Flags contains command-line flags
                Flags []ConfigurationFlag `json:"flags,omitempty"`
	}

	type ConfigurationList struct {
		TypeMeta `json:",inline"`
		ListMeta `json:"metadata,omitempty"`

		Items []Configuration `json:"items"`
	}
```

``Configuration`` information will be stored in ``etcd`` by default.

### Validation

``Flags`` are stored in the Configuration as a key value string array. They are stored in the same
form as command line flags. The ``Kubernetes`` component using the flags loads and validates
them as they would flags specified on the command line.

As on the command line, some flags may be specified more than once, some flags may require a value.

In cases where a flag has multiple names, the long form canonical name is the one that the
Configuration should store.

As on the command line, unknown flags will cause a validation failure.

### Creation and Modification

At present standard means for storing and updating items will be used with Configuration objects:

```
kubectl create -f configuration.json
kubectl replace ...
```

More detailed ```kubectl``` commands may be added that build on top of this later.

### Examples
Examples:
```json
{
    "apiVersion": "v1",
    "kind": "Configuration",
    "metadata": {
        "name": "scheduler"
    }
    "flags": [
        {
            "flag": "node-monitor-period",
	    "value": "10s"
        },
        {
	    "flag": "profiling",
            "value": "true"
        }
    ]
}
```

### Implementation Notes

Common component command line parsing code will be updated to load configuration from the API
server and watch for changes. In addition watching code should be implemented at this level,
making it trivial for components to use updated flags if they examine the value repeatedly.

A new command line argument to override the ``Configuration`` name to be used by a component
will be added to certain components (eg: ``kubelet``).


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/configuration.md?pixel)]()
