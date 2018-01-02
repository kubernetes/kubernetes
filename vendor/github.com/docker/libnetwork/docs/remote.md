Remote Drivers
==============

The `drivers.remote` package provides the integration point for dynamically-registered drivers. Unlike the other driver packages, it does not provide a single implementation of a driver; rather, it provides a proxy for remote driver processes, which are registered and communicate with LibNetwork via the Docker plugin package.

For the semantics of driver methods, which correspond to the protocol below, please see the [overall design](design.md).

## LibNetwork integration with the Docker `plugins` package

When LibNetwork initialises the `drivers.remote` package with the `Init()` function, it passes a `DriverCallback` as a parameter, which implements `RegisterDriver()`. The remote driver package uses this interface to register remote drivers with LibNetwork's `NetworkController`, by supplying it in a `plugins.Handle` callback.

The callback is invoked when a driver is loaded with the `plugins.Get` API call. How that comes about is out of scope here (but it might be, for instance, when that driver is mentioned by the user).

This design ensures that the details of driver registration mechanism are owned by the remote driver package, and it doesn't expose any of the driver layer to the North of LibNetwork.

## Implementation

The remote driver implementation uses a `plugins.Client` to communicate with the remote driver process. The `driverapi.Driver` methods are implemented as RPCs over the plugin client.

The payloads of these RPCs are mostly direct translations into JSON of the arguments given to the method. There are some exceptions to account for the use of the interfaces `InterfaceInfo` and `JoinInfo`, and data types that do not serialise to JSON well (e.g., `net.IPNet`). The protocol is detailed below under "Protocol".

## Usage

A remote driver proxy follows all the rules of any other in-built driver and has exactly the same `Driver` interface exposed. LibNetwork will also support driver-specific `options` and user-supplied `labels` which may influence the behaviour of a remote driver process.

## Protocol

The remote driver protocol is a set of RPCs, issued as HTTP POSTs with JSON payloads. The proxy issues requests, and the remote driver process is expected to respond usually with a JSON payload of its own, although in some cases these are empty maps.

### Errors

If the remote process cannot decode, or otherwise detects a syntactic problem with the HTTP request or payload, it must respond with an HTTP error status (4xx or 5xx).

If the remote process http server receives a request for an unknown URI, it should respond with the HTTP StatusCode `404 Not Found`. This allows LibNetwork to detect when a remote driver does not implement yet a newly added method, therefore not to deem the request as failed.

If the remote process can decode the request, but cannot complete the operation, it must send a response in the form

    {
		"Err": string
    }

The string value supplied may appear in logs, so should not include confidential information.

### Handshake

When loaded, a remote driver process receives an HTTP POST on the URL `/Plugin.Activate` with no payload. It must respond with a manifest of the form

    {
		"Implements": ["NetworkDriver"]
    }

Other entries in the list value are allowed; `"NetworkDriver"` indicates that the plugin should be registered with LibNetwork as a driver.

### Set capability

After Handshake, the remote driver will receive another POST message to the URL `/NetworkDriver.GetCapabilities` with no payload. The driver's response should have the form:

	{
		"Scope":             "local"
		"ConnectivityScope": "global"
	}

Value of "Scope" should be either "local" or "global" which indicates whether the resource allocations for this driver's network can be done only locally to the node or globally across the cluster of nodes. Any other value will fail driver's registration and return an error to the caller.
Similarly, value of "ConnectivityScope" should be either "local" or "global" which indicates whether the driver's network can provide connectivity only locally to this node or globally across the cluster of nodes. If the value is missing, libnetwork will set it to the value of "Scope". should be either "local" or "global" which indicates

### Create network

When the proxy is asked to create a network, the remote process shall receive a POST to the URL `/NetworkDriver.CreateNetwork` of the form

    {
		"NetworkID": string,
		"IPv4Data" : [
		{
			"AddressSpace": string,
			"Pool": ipv4-cidr-string,
			"Gateway" : ipv4-cidr-string,
			"AuxAddresses": {
				"<identifier1>" : "<ipv4-address1>",
				"<identifier2>" : "<ipv4-address2>",
				...
			}
		},
		],
		"IPv6Data" : [
		{
			"AddressSpace": string,
			"Pool": ipv6-cidr-string,
			"Gateway" : ipv6-cidr-string,
			"AuxAddresses": {
				"<identifier1>" : "<ipv6-address1>",
				"<identifier2>" : "<ipv6-address2>",
				...
			}
		},
		],
		"Options": {
			...
		}
    }

* `NetworkID` value is generated by LibNetwork which represents a unique network. 
* `Options` value is the arbitrary map given to the proxy by LibNetwork. 
* `IPv4Data` and `IPv6Data` are the ip-addressing data configured by the user and managed by IPAM driver. The network driver is expected to honor the ip-addressing data supplied by IPAM driver. The data include,
* `AddressSpace` : A unique string represents an isolated space for IP Addressing 
* `Pool` : A range of IP Addresses represented in CIDR format address/mask. Since, the IPAM driver is responsible for allocating container ip-addresses, the network driver can make use of this information for the network plumbing purposes.
* `Gateway` : Optionally, the IPAM driver may provide a Gateway IP address in CIDR format for the subnet represented by the Pool. The network driver can make use of this information for the network plumbing purposes.
* `AuxAddresses` : A list of pre-allocated ip-addresses with an associated identifier as provided by the user to assist network driver if it requires specific ip-addresses for its operation.

The response indicating success is empty:

    {}

### Delete network

When a network owned by the remote driver is deleted, the remote process shall receive a POST to the URL `/NetworkDriver.DeleteNetwork` of the form

    {
		"NetworkID": string
    }

The success response is empty:

    {}

### Create endpoint

When the proxy is asked to create an endpoint, the remote process shall receive a POST to the URL `/NetworkDriver.CreateEndpoint` of the form

    {
		"NetworkID": string,
		"EndpointID": string,
		"Options": {
			...
		},
		"Interface": {
			"Address": string,
			"AddressIPv6": string,
			"MacAddress": string
		}
    }

The `NetworkID` is the generated identifier for the network to which the endpoint belongs; the `EndpointID` is a generated identifier for the endpoint.

`Options` is an arbitrary map as supplied to the proxy.

The `Interface` value is of the form given. The fields in the `Interface` may be empty; and the `Interface` itself may be empty. If supplied, `Address` is an IPv4 address and subnet in CIDR notation; e.g., `"192.168.34.12/16"`. If supplied, `AddressIPv6` is an IPv6 address and subnet in CIDR notation. `MacAddress` is a MAC address as a string; e.g., `"6e:75:32:60:44:c9"`.

A success response is of the form

    {
		"Interface": {
			"Address": string,
			"AddressIPv6": string,
			"MacAddress": string
		}
    }

with values in the `Interface` as above. As far as the value of `Interface` is concerned, `MacAddress` and either or both of `Address` and `AddressIPv6` must be given.

If the remote process was supplied a non-empty value in `Interface`, it must respond with an empty `Interface` value. LibNetwork will treat it as an error if it supplies a non-empty value and receives a non-empty value back, and roll back the operation.

### Endpoint operational info

The proxy may be asked for "operational info" on an endpoint. When this happens, the remote process shall receive a POST to `/NetworkDriver.EndpointOperInfo` of the form

    {
		"NetworkID": string,
		"EndpointID": string
    }

where `NetworkID` and `EndpointID` have meanings as above. It must send a response of the form

    {
		"Value": { ... }
    }

where the value of the `Value` field is an arbitrary (possibly empty) map.

### Delete endpoint

When an endpoint is deleted, the remote process shall receive a POST to the URL `/NetworkDriver.DeleteEndpoint` with a body of the form

    {
		"NetworkID": string,
		"EndpointID": string
    }

where `NetworkID` and `EndpointID` have meanings as above. A success response is empty:

    {}

### Join

When a sandbox is given an endpoint, the remote process shall receive a POST to the URL `NetworkDriver.Join` of the form

    {
		"NetworkID": string,
		"EndpointID": string,
		"SandboxKey": string,
		"Options": { ... }
    }

The `NetworkID` and `EndpointID` have meanings as above. The `SandboxKey` identifies the sandbox. `Options` is an arbitrary map as supplied to the proxy.

The response must have the form

    {
		"InterfaceName": {
			SrcName: string,
			DstPrefix: string
		},
		"Gateway": string,
		"GatewayIPv6": string,
		"StaticRoutes": [{
			"Destination": string,
			"RouteType": int,
			"NextHop": string,
		}, ...]
    }

`Gateway` is optional and if supplied is an IP address as a string; e.g., `"192.168.0.1"`. `GatewayIPv6` is optional and if supplied is an IPv6 address as a string; e.g., `"fe80::7809:baff:fec6:7744"`.

The entries in `InterfaceName` represent actual OS level interfaces that should be moved by LibNetwork into the sandbox; the `SrcName` is the name of the OS level interface that the remote process created, and the `DstPrefix` is a prefix for the name the OS level interface should have after it has been moved into the sandbox (LibNetwork will append an index to make sure the actual name does not collide with others).

The entries in `"StaticRoutes"` represent routes that should be added to an interface once it has been moved into the sandbox. Since there may be zero or more routes for an interface, unlike the interface name they can be supplied in any order.

Routes are either given a `RouteType` of `0` and a value for `NextHop`; or, a `RouteType` of `1` and no value for `NextHop`, meaning a connected route.

If no gateway and no default static route is set by the driver in the Join response, LibNetwork will add an additional interface to the sandbox connecting to a default gateway network (a bridge network named *docker_gwbridge*) and program the default gateway into the sandbox accordingly, pointing to the interface address of the bridge *docker_gwbridge*.

### Leave

If the proxy is asked to remove an endpoint from a sandbox, the remote process shall receive a POST to the URL `/NetworkDriver.Leave` of the form

    {
		"NetworkID": string,
		"EndpointID": string
    }

where `NetworkID` and `EndpointID` have meanings as above. The success response is empty:

    {}

### DiscoverNew Notification

LibNetwork listens to inbuilt docker discovery notifications and passes it along to the interested drivers. 

When the proxy receives a DiscoverNew notification, the remote process shall receive a POST to the URL `/NetworkDriver.DiscoverNew` of the form

    {
		"DiscoveryType": int,
		"DiscoveryData": {
			...
		}
    }

`DiscoveryType` represents the discovery type. Each Discovery Type is represented by a number.
`DiscoveryData` carries discovery data the structure of which is determined by the DiscoveryType

The response indicating success is empty:

    {}

*  Node Discovery

Node Discovery is represented by a `DiscoveryType` value of `1` and the corresponding `DiscoveryData` will carry Node discovery data.

    {
		"DiscoveryType": int,
		"DiscoveryData": {
                    "Address" : string
                    "self" : bool
		}
    }

### DiscoverDelete Notification

When the proxy receives a DiscoverDelete notification, the remote process shall receive a POST to the URL `/NetworkDriver.DiscoverDelete` of the form

    {
		"DiscoveryType": int,
		"DiscoveryData": {
			...
		}
    }

`DiscoveryType` represents the discovery type. Each Discovery Type is represented by a number.
`DiscoveryData` carries discovery data the structure of which is determined by the DiscoveryType

The response indicating success is empty:

    {}

* Node Discovery

Similar to the DiscoverNew call, Node Discovery is represented by a `DiscoveryType` value of `1` and the corresponding `DiscoveryData` will carry Node discovery data to be deleted.

    {
		"DiscoveryType": int,
		"DiscoveryData": {
                    "Address" : string
                    "self" : bool
		}
    }
