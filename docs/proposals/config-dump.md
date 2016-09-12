<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should refer to the docs that go with that version.

Documentation for other releases can be found at [releases.K8s.io](http://releases.K8s.io).
</strong>

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->


# Overview

Here we describe a proposal for retrieving configurations (configs) that govern the behavior of any one of the Kubernetes (K8s) components. By components we mean any of the following: apiserver, scheduler, control-manager, and kubelet. We expect that these configurations be for the most part static through the life of the component but fully understand that changes from specs may affect these. Consequently, any retrieval of the configurations is to be considered true and valid at the time of retrieval and we will assume that the user is well aware of any transient behavior.   

We propose several important changes:
* new kubectl command: 
    * `kubectl get componentconfigs <componentName> (alias compcs) --id <componentId>`
* adding new API: 
    * `/api/alpha/component/name/<componentName>/configurations?id=<componentId>` \*
* creating maintainable configuration objects to components
* adding api endpoints to individual components to request configs directly, e.g., in the case of a standalone node running kubelet
    * `/api/alpha/configurations` 

\* Query parameter id will optionally allow you to select a particular component in the case where there may be a set of them, such as in the case of kubelet. Where id is left out and we have a set of components we should expect to produce a list of configs results. 

## Motivation

By necessity Kubernetes is a stateful thing made up of components who have been configured in a particular way. Currently the K8s code base, as 09/2016, does not support a way nor means by which to query for the set of configurations that led to the current state of any component. As a consequence it is extremely difficult to debug the current state of a K8s cluster, especially when original configurations have been lost, or are simply not available. 

## Design

### API changes
New endpoints in the api are to be created: 

```
/api/alpha/component/name/<componentName>/configurations?id=<componentId>
```

supporting the following HTTP method

``` 
GET - retrieve component configurations. **
```

\*\* The `PUT` method is to be discussed in the [Future work](#Future work) section of this proposal.

An interface should be created at the component level that allows different classifications of configurations to be added to a particular component. The point being that the final configuration object is composed of a multitude classes of very specific classifications presumably defined by interested parties.

Example request of a cluster with one running apiserver component:

```
kubectl get componentconfigs apiserver
```

for which the api call looks like:

```
/api/alpha/component/name/apiserver/configurations
```

both with the example output:

``` 
{
    "kind": "component-configs",
    "apiVersion": "alpha",
    "metadata":{
        "component": "apiserver",
        "id": "718G918A-D053-EA22-BC8B-EIA8363BA071"
        "timeStamp": "2016-09-13T14:47:35+00:00",
        "error": []
    },
    "configurations": {
        "flags": {
            "cloud-provider": "gce",
            ...,
        },
        "system": {
            "" : ""
        }
    }
}
```

We added error to metadata for the component-configs object in the case we fail to receive any configurations from a particular component. We propose that errors would look like:

```
"error": [
    {
        "statusCode": 408,
        "configurationName": "flags"
        "message": "timed out"
    }
]
```
 
where the message and status code represent the error received at the time of request failure. We would like that a failure to retrieve a particular configurationName would result on its own failure but not block the retrieval of those that succeeded, this probably could be the focus of a future implementation and may be beyond our current scope.

### Component API changes

Some components (kubelet) have endpoints that can be altered to include an endpoint such as

```
 /api/alpha/configurations
```

where the expected output is:

```
{
    "kind": "component-configs",
    "apiVersion": "alpha",
    "metadata": {
        "component": "kubelet",
        "id": "818B908B-D053-CB11-BC8B-EEA826EBA090"
        "timeStamp": "2016-09-13T14:47:35+00:00",
        "error": []
    },
    "configurations": {
        "flags": {
            "cloud-provider": "aws",
            ...
        },
        "system": {
            "someKey" : "someValue",
            ...
        }
    }
}
```

Components that do not support such endpoints will most likely have their configurations captured at some point before they are deployed. Eventing system may be needed, to capture changes but that is left for future considerations and out of the scope of current discussion.

### Interfaces and types 

The component configurations for this proposal will simply retrieve configurations and will support changes at a later date, the proposed interfaces here will reflect this choice.

First we describe the high level fetching of configurations as reported by components:

```
type ConfigurationRetriever interface {
    // these should be retrieved and potentially cached for a short time.
    Retrieve(component string) ComponentConfigs
}
```

It will be assumed that any communication with an unresponsive components will result in an error code.

Each component will have a map that represents the classification of configurations, each should be added to the component's configurations.

```
type ComponentConfigs interface {
    // return the map of configurations
    GetConfigurations() map[string]ComponentConfigurator

    // add a detailed component configuration type
    AppendConfiguration(name string, configs ComponentConfigurator)
}

```

ComponentConfigurator is the detailed configuration interface that should be implemented by a self contained classification (struct), such as Flags.

```
type ComponentConfigurator interface {
    // retrieve a configuration
    retrieve(key string) string

    // retrieve the map of configurations
    retrieveAll() map[string]string
}
```

We recommend caution in never be exposing sensitive data by any implementation of this interface.

## Future work

The current api work will extract configurations at the component level. One additional *would like* feature would be to extract all configs system wide at once. We expect this to be immediately done after extraction of the configurations are complete for the component level. We suggest that the api call to look like

```
/api/alpha/component/all/configurations
```

Additionally we would like to support the change/update some configurations which would require the a white-list of items that could change:

```
PUT /api/alpha/component/name/apiserver/configurations  
```

with its body being

```
{
    "flags": {
        "someKey": "someValue",
        ...        
    },
    "someConfig": {
        "someKey": "someValue",
        ...
    }
}
```

where the type of configuration is defined. An error should occur if value is not allowed to change in run-time. In addition we can consider the api above 

```
PUT /api/alpha/component/name/apiserver/configurations/<configName>  
```

and for example:

```
PUT /api/alpha/component/name/apiserver/configurations/flags
```

with its body:

```
{
    "someKey": "someValue",
    ...
}
```



<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/runtimeconfig.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

