<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<strong>
The latest 1.0.x release of this document can be found
[here](http://releases.k8s.io/release-1.0/docs/http-cloud-provider.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# HTTP Cloudprovider

# Motivation

Currently, the cloudproviders are designed so that they get directly compiled into kubernetes. It is very hard to add custom cloudproviders to the kubernetes platform without using a custom build of kubernetes. Therefore, we would like to build an HTTP Cloudprovider which would be able to send a HTTP request by converting the cloudprovider interfaces to an HTTP API. The cloudprovider would need an HTTP endpoint so that kubernetes would be able to send a request to them and then the cloudproviders would be able to send the desired service information back to kubernetes.

# Specification

The user should specify a config file in JSON format. The **clientURL** field is required which specifies the URL and port of the cloud provider.

Instances are grouped by locale (region or zone which is depended on Cloudproviders' supports). When this module is init'ed, it call GetZones() to get the its locale and is used for subsequent calls

##### Example Config File:

```json
{
    "clientURL" :"http://127.0.0.1:8080"
}
```

First the API checks for which interfaces in [cloud.go](../pkg/cloudprovider/cloud.go) are implemented.

A OPTIONS request for /cloudprovider/v1alpha1/{Interface} will return whether the interface is supported (200 Response Code) or not (501 Not Implemented Response Code).

The following methods must be implemented by the HTTP cloudprovider.

If you encounter an error in processing the request, return an HTTP 404 with a response body matching [api.Status](https://github.com/kubernetes/kubernetes/blob/master/pkg/api/v1/types.go#L1586-L1607)


## Provider Name

#### GET /cloudprovider/v1alpha1/providerName

Returns the cloud provider name.

##### Expected return value:

|Name|Description|Example Value|
|----|-----------|-------------|
|providerName|Name of the cloud provider.|"my-cloud"|
Example

```json
{
    "providerName": "my-cloud"
}
```

## Instances

The instances interface allows the kubernetes platform to query the cloud provider about the instances currently within the cloud provider system. Instances are grouped by locale (Some cloudproviders support 'region' others support 'zone')

### Methods

#### OPTIONS /cloudprovider/v1alpha1/instances

Returns whether the Instance interface is supported or not.

##### Expected return value:

Empty 200 Response if supported else 501 Response


#### GET /cloudprovider/v1alpha1/locales/{locale}/instances/{instanceName}
Returns an instance for the given instance name.

##### Parameters

|Type|Name|Description|Required|Schema|Default
|----|----|-----------|--------|------|-------
|PathParameter|name|name of the instance|true|string

##### Responses

|HTTP Code|Description|Schema|
|---------|-----------|------|
|200|success|Instance|

Example

```json
{
    "metadata": {
        "name": "instance-1",
    },
    "spec": {
        "instanceID":"instanceID01",
        "nodeAddress":[
            {
                "type": "InternalIP",
                "address": "127.0.0.1",
            },
        ]
    }
}
```

#### GET /cloudprovider/v1alpha1/locales/{locale}/instances/{FQDN}

Returns all instances where the name matches the word FQDN. FQDN is a go regular expression.

##### Expected return value:

##### Parameters

|Type|Name|Description|Required|Schema|Default
|----|----|-----------|--------|------|-------

##### Responses

|HTTP Code|Description|Schema|
|---------|-----------|------|
|200|success|InstanceList|

```json
{
    "items": [
        {
            "metadata": {
                "name": "instance-1",
            },
            "spec": {
                "instanceID":"instanceID01",
                "nodeAddress":[
                    {
                        "type": "InternalIP",
                        "address": "127.0.0.1"
                    },
                ]
            },
        },
    ]
}
```

#### GET /cloudprovider/v1alpha1/locales/{locale}/hostnames/{hostName}

Returns the name of the node which is related to the specific host.

##### Parameters

|Type|Name|Description|Required|Schema|Default
|----|----|-----------|--------|------|-------

##### Responses

|HTTP Code|Description|Schema|
|---------|-----------|------|
|200|success|Instance|


#### POST /cloudprovider/v1alpha1/locales/{locale}/SSHKeyToAll

Adds the SSH Key provided to all the instances. Returns whether the key was added or not.

##### Parameters

|Type|Name|Description|Required|Schema|Default
|----|----|-----------|--------|------|-------
|BodyParameter|body|SSH Key schema|true|InstanceSSHKey

Example
keydata: The actual ssh key. Any format allowed not restricted to strings.
```json
{
    "user": "name",
    "keyData": "zPjoihsswRTGIUHKLHIHO345@435"
}
```

##### Responses

|HTTP Code|Description|Schema|
|---------|-----------|------|
|200|success|api.Status|


## TCPLoadBalancers

The TCPLoadBalancers interface allows the kubernetes platform to query the cloud provider about the Load Balancer the cloud provider has and requests to change the load balancer as required.

### Methods

#### OPTIONS /cloudprovider/v1alpha1/tcpLoadBalancers

Know whether the interface is supported or not.

##### Expected return value:

Empty 200 Response if supported else 501 Response

#### GET /cloudprovider/v1alpha1/locale/{locale}/tcpLoadBalancers/{name}

Returns the TCP Load Balancer Status if it exists in the region with the particular name.

##### Parameters

|Type|Name|Description|Required|Schema|Default
|----|----|-----------|--------|------|-------
|PathParameter|name|name of the tcpLoadBalancer|true|string

##### Responses

|HTTP Code|Description|Schema|
|---------|-----------|------|
|200|success|TCPLoadBalancer|

Example

```json
{
    "metadata": {
        "name": "tcploadbalancer-1",
    },
    "spec":{
        "region": "west",
        "externalIP": "1.2.3.4",
        "ports": [ 
            {
                "name":"SMTP",
                "protocol":"TCP",
                "port":1234,
                "targetPort":1234,
                "nodePort":1234
            },
        ],
        "hosts": [ "host1", "host2"  ],
        "sessionAffinity":"None"
    },
    "status": {
        "ingress": [
            {
                "ip": "127.0.0.1",
                "hostname": "my-cloud-host1",
            },
        ],
        "exists": true
    }
}
```

#### POST /cloudprovider/v1alpha1/locales/{locale}/tcpLoadBalancers

Creates a new tcp load balancer and return the status of the balancer.

##### Parameters

|Type|Name|Description|Required|Schema|Default
|----|----|-----------|--------|------|-------
|BodyParameter|name|a TCPLoadBalancer schema|true|TCPLoadBalancer

##### Responses

|HTTP Code|Description|Schema|
|---------|-----------|------|
|200|success|TCPLoadBalancer|

#### PUT /cloudprovider/v1alpha1/locales/{locale}/tcpLoadBalancers/{name}

Update hosts under the specified tcp load balancer

##### Parameters

|Type|Name|Description|Required|Schema|Default
|----|----|-----------|--------|------|-------
|PathParameter|name|a TCPLoadBalancer schema|true|TCPLoadBalancer

##### Responses

|HTTP Code|Description|Schema|
|---------|-----------|------|
|200|success|api.Status|

#### DELETE /cloudprovider/v1alpha1/locales/{locale}/tcpLoadBalancers/{name}

Deletes the specified Load Balancer.

##### Parameters

|Type|Name|Description|Required|Schema|Default
|----|----|-----------|--------|------|-------
|PathParameter|name|name of the tcpLoadBalancer|true|string

##### Responses

|HTTP Code|Description|Schema|
|---------|-----------|------|
|200|success|api.Status|


## Zones

The Zones interface allows the kubernetes platform to query the cloud provider about the Zones and which areas are having a failed state.

### Methods

#### OPTIONS /cloudprovider/v1alpha1/zones

Know whether the interface is supported or not.

##### Expected return value:

Empty 200 Response if supported else 501 Response

#### GET /cloudprovider/v1alpha1/zones

Returns the Zone containing the current failure zone and locality region that the program is running in.

##### Parameters

|Type|Name|Description|Required|Schema|Default
|----|----|-----------|--------|------|-------

##### Responses

|HTTP Code|Description|Schema|
|---------|-----------|------|
|200|success|Zone|

Example

```json
{
    "metadata": {
        "name": "zone-west",
    },
    "spec": {
        "failureDomain":"my-zone-2",
        "region": "west",
    },
}
```

## Clusters

The clusters interface allows the kubernetes platform to query information about the clusters inside the cloud provider.

### Methods

#### OPTIONS /cloudprovider/v1alpha1/clusters

Know whether the interface is supported or not.

##### Expected return value:

Empty 200 Response if supported else 501 Response

#### GET /cloudprovider/v1alpha1/locales/{locale}/clusters

Returns a list of all clusters.

##### Parameters

|Type|Name|Description|Required|Schema|Default
|----|----|-----------|--------|------|-------

##### Responses

|HTTP Code|Description|Schema|
|---------|-----------|------|
|200|success|ClusterList|

Example

```json
{
    "items": [
        {
            "metadata": {
                "name": "my-cluster",
            },
            "spec": {
                "masterAddress": "1.2.3.4"
            },
        },
    ],
}
```

#### GET /cloudprovider/v1alpha1/locales/{locale}/clusters/{clusterName}

Returns the address of the master of the cluster provided.

##### Parameters

|Type|Name|Description|Required|Schema|Default
|----|----|-----------|--------|------|-------
|PathParameter|name|a cluster name|true|string

##### Responses

|HTTP Code|Description|Schema|
|---------|-----------|------|
|200|success|Cluster|

Example

```json
{
    "metadata": {
        "name": "my-cluster",
    },
    "spec": {
        "masterAddress": "1.2.3.4"
    },
}
```

## Routes

The routes interface allows the kubernetes platform to query information about the routes and change them inside the cloud provider.

### Methods

#### OPTIONS /cloudprovider/v1alpha1/routes

Know whether the interface is supported or not.

##### Expected return value:

Empty 200 Response if supported else 501 Response

#### GET /cloudprovider/v1alpha1/locales/{locale}/clusterNames/{clusterName}/routes

Returns a list of all the routes belonging to the specified cluster.

##### Parameters

|Type|Name|Description|Required|Schema|Default
|----|----|-----------|--------|------|-------

##### Responses

|HTTP Code|Description|Schema|
|---------|-----------|------|
|200|success|RouteList|

Example

```json
{
    "items": [
        {
            "metadata": {
                "name": "route-01",
            },
            "spec": {
                "nameHint": "a hint"
                "targetInstance": "a1-small",
                "destinationCIDR": "192.168.1.0/24"
            },
        },
    ],
}
```

#### POST /cloudprovider/v1alpha1/locales/{locale}/clusterNames/{clusterName}/routes

Creates a route inside the cloud provider and returns whether the route was created or not.

##### Parameters

|Type|Name|Description|Required|Schema|Default
|----|----|-----------|--------|------|-------
|BodyParameter|name|a Route schema|true|Route

##### Responses

|HTTP Code|Description|Schema|
|---------|-----------|------|
|200|success|api.Status|


#### DELETE /cloudprovider/v1alpha1/locales/{locale}/clusterNames/{clusterName}/routes/{routeName}

Delete the requested route within the specified cluster.

##### Parameters

|Type|Name|Description|Required|Schema|Default
|----|----|-----------|--------|------|-------
|PathParameter|name|name of the route|true|string

##### Responses

|HTTP Code|Description|Schema|
|---------|-----------|------|
|200|success|api.Status|


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/http-cloud-provider.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
