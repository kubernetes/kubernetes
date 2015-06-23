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

##### Example Config File:

```json
{
    "clientURL" :"http://127.0.0.1:8080"
}
```

First the API checks for which interfaces in [cloud.go](../pkg/cloudprovider/cloud.go) are implemented.

A OPTIONS request for /cloudprovider/v1aplha1/{Interface} will return whether the interface is supported (200 Response Code) or not (501 Not Implemented Response Code).

The following methods must be implemented by the HTTP cloudprovider.

If you encounter an error in processing the request, return an HTTP 404 with a response body matching [api.Status](https://github.com/kubernetes/kubernetes/blob/master/pkg/api/v1/types.go#L1586-L1607)


## Provider Name

#### GET /cloudprovider/v1aplha1/providerName

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

The instances interface allows the kubernetes platform to query the cloud provider about the instances currently within the cloud provider system.

### Methods

#### OPTIONS /cloudprovider/v1aplha1/instances

Returns whether the Instance interface is supported or not.

##### Expected return value:

Empty 200 Response if supported else 501 Response


#### GET /cloudprovider/v1aplha1/instances/{instanceName}/nodeAddresses

Returns all node addresses for the given instance.

##### Expected return value:

|Name|Description|Example Value|
|----|-----------|-------------|
|nodeAddresses|List of node adress objects.||
|nodeAddresses:type|The IP type relate to the node address. The types of addresses are restricted to the list below.|"InternalIP"|
|nodeAddresses:address|The actual node's address.|"127.0.0.1"|

Address Types

|Name|Description|
|----|-----------|
|InternalIP|The node's Internal IP address.|
|ExternalIP|The node's External IP address.|
|Hostname|The hostname of the node.|
|LegacyHostIP|The node's Host IP address.|
Example

```json
{
    "nodeAddresses": [
        {
            "type": "InternalIP",
            "address": "127.0.0.1"
        }
    ]
}
```

#### GET /cloudprovider/v1aplha1/instances/{instanceName}/ID

Returns the Instance ID for the given instance.

##### Expected return value:

|Name|Description|Example Value|
|----|-----------|-------------|
|instanceID|ID of the cloud provider.|"my-cloud-name1"|
Example

```json
{
    "instanceID": "my-cloud-name1"
}
```

#### GET /cloudprovider/v1aplha1/instances/{FQDN}

Returns all instances where the name matches the word FQDN. FQDN is a go regular expression.

##### Expected return value:

|Name|Description|Example Value|
|----|-----------|-------------|
|instances|List of all instances matching the FQDN.||
|instances:instanceName|Name of the instance.|"cloud-1"|
Example

```json
{
    "instances": [
        {
            "instanceName": "cloud-1"
        }
    ]
}
```


#### GET /cloudprovider/v1aplha1/instances/node/{hostName}

Returns the name of the node which is related to the specific host.

##### Expected return value:

|Name|Description|Example Value|
|----|-----------|-------------|
|nodeName|The name of the node in the host|"my-node"|
Example

```json
{
    "nodeName": "my-node"
}
```

#### POST /cloudprovider/v1aplha1/instances/sshKey

Adds the SSH Key provided to all the instances. Returns whether the key was added or not.

##### Sender body:

|Name|Description|Example Value|
|----|-----------|-------------|
|user|The user's whose SSH key needs to be added.|"name"|
|keydata|The actual ssh key. Any format allowed not restricted to strings.|"zPjoihsswRTGIUHKLHIHO345@435"|
Example

```json
{
    "user": "name",
    "keyData": "zPjoihsswRTGIUHKLHIHO345@435"
}
```

##### Expected Return Value:

|Name|Description|Example Value|
|----|-----------|-------------|
|SSHKeyAdded|Boolean value regarding whether the SSH Key was added or not.|false|
Example

```json
{
    "SSHKeyAdded": false
}
```

## TCPLoadBalancers

The TCPLoadBalancers interface allows the kubernetes platform to query the cloud provider about the Load Balancer the cloud provider has and requests to change the load balancer as required.

### Methods

#### OPTIONS /cloudprovider/v1aplha1/tcpLoadBalancers

Know whether the interface is supported or not.

##### Expected return value:

Empty 200 Response if supported else 501 Response

#### GET /cloudprovider/v1aplha1/tcpLoadBalancers/{region}/{name}

Returns the TCP Load Balancer Status if it exists in the region with the particular name.

##### Expected return value:

|Name|Description|Example Value|
|----|-----------|-------------|
|ingress|List of ingress points for the load balancer.||
|ingress:ip|The IP address of the ingress point. This is a optional field as only one of IP or hostname is required.|"127.0.0.1"|
|ingress:hostname|The hostname of the ingress point. This is a optional field as only one of IP or hostname is required.|"my-cloud-host1"|
|exists|Whether the TCP Load Balancer exists or not.|true|
Example

```json
{
    "ingress": [
        {
            "ip":"127.0.0.1",
            "hostname":"my-cloud-host1"
        }
    ],
    "exists":true
}
```

#### POST /cloudprovider/v1aplha1/tcpLoadBalancers

Creates a new tcp load balancer and return the status of the balancer.

##### Sender body:

|Name|Description|Example Value|
|----|-----------|-------------|
|loadBalancerName|The name of the TCP load balancer.|"my-tcp-balancer"|
|region|The location of the load balancer.|"USA"|
|externalIP|The external IP of the load balancer.|"127.0.0.1"|
|ports|List of all the service port objects.||
|ports:name|Name of the port. If only one port is present then this field is **optional**.|"SMTP"|
|ports:protocol|The protocol the port works on. The value can only be either "UDP" or "TCP".|"TCP"|
|ports:port|The port that will be exposed on the service.|1234|
|ports:targetPort|The target port on pods selected by this service. This field is optional. It can be **string** or **int** type. If not provided, will use **ports:port** value.|1234|
|ports:nodePort|The port on each node on which this service is exposed.|1234|
|hosts|A list of all host names.||
|hosts:hostname|A host name.|"my-cloud-host1"|
|sessionAffinity|The session affinity. The value can only be either "None" or "ClientIP".|"None"|
Example

```json
{
    "loadBalancerName":"my-tcp-balancer",
    "region":"USA",
    "externalIP":"127.0.0.1",
    "ports":[
        {
            "name":"SMTP",
            "protocol":"TCP",
            "port":1234,
            "targetPort":1234,
            "nodePort":1234
        }
    ],
    "hosts":[
        {
            "hostname":"my-cloud-host1"
        }
    ],
    "sessionAffinity":"None"
}
```

##### Expected return value:

|Name|Description|Example Value|
|----|-----------|-------------|
|ingress|List of ingress points for the load balancer.||
|ingress:ip|The IP address of the ingress point. This is a optional field as only one of IP or hostname is required.|"127.0.0.1"|
|ingress:hostname|The hostname of the ingress point. This is a optional field as only one of IP or hostname is required.|"my-cloud-host1"|
|exists|Whether the TCP Load Balancer exists or not.|true|
Example

```json
{
    "ingress": [
        {
            "ip":"127.0.0.1",
            "hostname":"my-cloud-host1"
        }
    ]
}
```

#### PUT /cloudprovider/v1aplha1/tcpLoadBalancers/{region}/{name}

Updates the hosts given in the Load Balancer specified.

##### Sender body:

|Name|Description|Example Value|
|----|-----------|-------------|
|hosts|A list of all host names.||
|hosts:hostname|A host name.|"my-cloud-host1"|
Example

```json
{
    "hosts":[
        {
            "hostname":"my-cloud-host1"
        }
    ]
}
```

##### Expected return body:

Send back an empty 204 response if the object was updated successfully.

#### DELETE /cloudprovider/v1aplha1/tcpLoadBalancers/{region}/{name}

Deletes the specified Load Balancer.

##### Expected return value:

Send back an empty 204 response if the object was deleted successfully or it didn't exist before.

## Zones

The Zones interface allows the kubernetes platform to query the cloud provider about the Zones and which areas are having a failed state.

### Methods

#### OPTIONS /cloudprovider/v1aplha1/zones

Know whether the interface is supported or not.

##### Expected return value:

Empty 200 Response if supported else 501 Response

#### GET /cloudprovider/v1aplha1/zones

Returns the Zone containing the current failure zone and locality region that the program is running in.

##### Expected return value:

|Name|Description|Example Value|
|----|-----------|-------------|
|failureDomain|The failure zone.|"my-zone-2"|
|region|Locality region that the program is running in.|"USA"|
Example

```json
{
    "failureDomain":"my-zone-2",
    "region":"USA"
}
```

## Clusters

The clusters interface allows the kubernetes platform to query information about the clusters inside the cloud provider.

### Methods

#### OPTIONS /cloudprovider/v1aplha1/clusters

Know whether the interface is supported or not.

##### Expected return value:

Empty 200 Response if supported else 501 Response

#### GET /cloudprovider/v1aplha1/clusters

Returns a list of all clusters.

##### Expected return value:

|Name|Description|Example Value|
|----|-----------|-------------|
|clusters|The list of all cluster objects.||
|clusters:clusterName|The name of the cluster.|"my-cluster"|
Example

```json
{
    "clusters": [
        {
            "clusterName":"my-cluster"
        }
    ]
}
```

#### GET /cloudprovider/v1aplha1/clusters/{clusterName}/master

Returns the address of the master of the cluster provided.

##### Expected return value:

|Name|Description|Example Value|
|----|-----------|-------------|
|masterAddress|The address of the master in the cluster. Can be a DNS Hostname or an IP Address.|"127.0.0.1"|
Example

```json
{
    "masterAddress":"127.0.0.1"
}
```

## Routes

The routes interface allows the kubernetes platform to query information about the routes and change them inside the cloud provider.

### Methods

#### OPTIONS /cloudprovider/v1aplha1/routes

Know whether the interface is supported or not.

##### Expected return value:

Empty 200 Response if supported else 501 Response

#### GET /cloudprovider/v1aplha1/routes/{clusterName}

Returns a list of all the routes belonging to the specified cluster.

##### Expected return value:

|Name|Description|Example Value|
|----|-----------|-------------|
|routes|The list of route objects.||
|routes:routeName|The name of the routing rule.|"abc"|
|routes:targetInstance|The name of the instance as specified in routing rules for the cloud-provider.|"a1-small"|
|routes:destinationCIDR|The CIDR format IP range that this routing rule applies to.|"192.168.1.0/24"|
Example

```json
{
    "routes": [
        {
            "routeName":"abc",
            "targetInstance":"a1-small",
            "destinationCIDR":"192.168.1.0/24"
        }
    ]
}
```

#### POST /cloudprovider/v1aplha1/routes

Creates a route inside the cloud provider and returns whether the route was created or not.

##### Sender Body:

|Name|Description|Example Value|
|----|-----------|-------------|
|clusterName|The name of the cluster in which the route needs to be created.|"my-cluster"|
|nameHint|A more meaningful use of the route name created.|"hint"|
|route|The route object which needs to be created||
|route:routeName|The name of the routing rule.|"abc"|
|route:targetInstance|The name of the instance as specified in routing rules for the cloud-provider.|"a1-small"|
|route:destinationCIDR|The CIDR format IP range that this routing rule applies to.|"192.168.1.0/24"|
Example

```json
{
    "clusterName":"my-cluster",
    "nameHint":"hint",
    "route":
    {
        "routeName":"abc",
        "targetInstance":"a1-small",
        "destinationCIDR":"192.168.1.0/24"
    }
}
```

##### Expected Return Body:

|Name|Description|Example Value|
|----|-----------|-------------|
|routeCreated|Whether the route was created or not.|true|
Example

```json
{
    "routeCreated":true
}
```

#### DELETE /cloudprovider/v1aplha1/routes/{clusterName}/{routeName}

Delete the requested route within the specified cluster.

##### Expected return value:

Send back an empty 204 response if the object was deleted successfully


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/http-cloud-provider.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
