# IPAM Driver

During the Network and Endpoints lifecycle, the CNM model controls the IP address assignment for network and endpoint interfaces via the IPAM driver(s).
Libnetwork has a default, built-in IPAM driver and allows third party IPAM drivers to be dynamically plugged. On network creation, the user can specify which IPAM driver libnetwork needs to use for the network's IP address management. This document explains the APIs with which the IPAM driver needs to comply, and the corresponding HTTPS request/response body relevant for remote drivers.


## Remote IPAM driver

On the same line of remote network driver registration (see [remote.md](./remote.md) for more details), libnetwork initializes the `ipams.remote` package with the `Init()` function. It passes a `ipamapi.Callback` as a parameter, which implements `RegisterIpamDriver()`. The remote driver package uses this interface to register remote drivers with libnetwork's `NetworkController`, by supplying it in a `plugins.Handle` callback.  The remote drivers register and communicate with libnetwork via the Docker plugin package. The `ipams.remote` provides the proxy for the remote driver processes.


## Protocol

Communication protocol is the same as the remote network driver.

## Handshake

During driver registration, libnetwork will query the remote driver about the default local and global address spaces strings, and about the driver capabilities.
More detailed information can be found in the respective section in this document.

## Datastore Requirements

It is the remote driver's responsibility to manage its database. 

## Ipam Contract

The remote IPAM driver must serve the following requests:

- **GetDefaultAddressSpaces**

- **RequestPool**

- **ReleasePool**

- **Request address**

- **Release address**


The following sections explain each of the above requests' semantic, when they are called during network/endpoint lifecycle, and the corresponding payload for remote driver HTTP request/responses.


## IPAM Configuration and flow

A libnetwork user can provide IPAM related configuration when creating a network, via the `NetworkOptionIpam` setter function. 

```go
func NetworkOptionIpam(ipamDriver string, addrSpace string, ipV4 []*IpamConf, ipV6 []*IpamConf, opts map[string]string) NetworkOption
```

The caller has to provide the IPAM driver name and may provide the address space and a list of `IpamConf` structures for IPv4 and a list for IPv6. The IPAM driver name is the only mandatory field. If not provided, network creation will fail.

In the list of configurations, each element has the following form:

```go
// IpamConf contains all the ipam related configurations for a network
type IpamConf struct {
	// The master address pool for containers and network interfaces
	PreferredPool string
	// A subset of the master pool. If specified,
	// this becomes the container pool
	SubPool string
	// Input options for IPAM Driver (optional)
	Options map[string]string
	// Preferred Network Gateway address (optional)
	Gateway string
	// Auxiliary addresses for network driver. Must be within the master pool.
	// libnetwork will reserve them if they fall into the container pool
	AuxAddresses map[string]string
}
```

On network creation, libnetwork will iterate the list and perform the following requests to the IPAM driver:

1. Request the address pool and pass the options along via `RequestPool()`.
2. Request the network gateway address if specified. Otherwise request any address from the pool to be used as network gateway. This is done via `RequestAddress()`.
3. Request each of the specified auxiliary addresses via `RequestAddress()`.

If the list of IPv4 configurations is empty, libnetwork will automatically add one empty `IpamConf` structure. This will cause libnetwork to request IPAM driver an IPv4 address pool of the driver's choice on the configured address space, if specified, or on the IPAM driver default address space otherwise. If the IPAM driver is not able to provide an address pool, network creation will fail.
If the list of IPv6 configurations is empty, libnetwork will not take any action.
The data retrieved from the IPAM driver during the execution of point 1) to 3) will be stored in the network structure as a list of `IpamInfo` structures for IPv4 and a list for IPv6.

On endpoint creation, libnetwork will iterate over the list of configs and perform the following operation:

1. Request an IPv4 address from the IPv4 pool and assign it to the endpoint interface IPv4 address. If successful, stop iterating.
2. Request an IPv6 address from the IPv6 pool (if exists) and assign it to the endpoint interface IPv6 address. If successful, stop iterating.

Endpoint creation will fail if any of the above operation does not succeed

On endpoint deletion, libnetwork will perform the following operations:

1. Release the endpoint interface IPv4 address
2. Release the endpoint interface IPv6 address if present

On network deletion, libnetwork will iterate the list of `IpamData` structures and perform the following requests to ipam driver:

1. Release the network gateway address via `ReleaseAddress()`
2. Release each of the auxiliary addresses via `ReleaseAddress()`
3. Release the pool via `ReleasePool()`

### GetDefaultAddressSpaces

GetDefaultAddressSpaces returns the default local and global address space names for this IPAM. An address space is a set of non-overlapping address pools isolated from other address spaces' pools. In other words, same pool can exist on N different address spaces. An address space naturally maps to a tenant name. 
In libnetwork, the meaning associated to `local` or `global` address space is that a local address space doesn't need to get synchronized across the
cluster whereas the global address spaces does. Unless specified otherwise in the IPAM configuration, libnetwork will request address pools from the default local or default global address space based on the scope of the network being created. For example, if not specified otherwise in the configuration, libnetwork will request address pool from the default local address space for a bridge network, whereas from the default global address space for an overlay network.

During registration, the remote driver will receive a POST message to the URL `/IpamDriver.GetDefaultAddressSpaces` with no payload. The driver's response should have the form:


	{
		"LocalDefaultAddressSpace": string
		"GlobalDefaultAddressSpace": string
	}



### RequestPool

This API is for registering an address pool with the IPAM driver. Multiple identical calls must return the same result.
It is the IPAM driver's responsibility to keep a reference count for the pool.

```go
RequestPool(addressSpace, pool, subPool string, options map[string]string, v6 bool) (string, *net.IPNet, map[string]string, error)
```


For this API, the remote driver will receive a POST message to the URL `/IpamDriver.RequestPool` with the following payload:

    {
		"AddressSpace": string
		"Pool":         string 
		"SubPool":      string 
		"Options":      map[string]string 
		"V6":           bool 
    }

    
Where:

    * `AddressSpace` the IP address space. It denotes a set of non-overlapping pools.
    * `Pool` The IPv4 or IPv6 address pool in CIDR format
    * `SubPool` An optional subset of the address pool, an ip range in CIDR format
    * `Options` A map of IPAM driver specific options
    * `V6` Whether an IPAM self-chosen pool should be IPv6
    
AddressSpace is the only mandatory field. If no `Pool` is specified IPAM driver may choose to return a self chosen address pool. In such case, `V6` flag must be set if caller wants an IPAM-chosen IPv6 pool. A request with empty `Pool` and non-empty `SubPool` should be rejected as invalid.
If a `Pool` is not specified IPAM will allocate one of the default pools. When `Pool` is not specified, the `V6` flag should be set if the network needs IPv6 addresses to be allocated.

A successful response is in the form:


	{
		"PoolID": string
		"Pool":   string
		"Data":   map[string]string
	}


Where:

* `PoolID` is an identifier for this pool. Same pools must have same pool id.
* `Pool` is the pool in CIDR format
* `Data` is the IPAM driver supplied metadata for this pool


### ReleasePool

This API is for releasing a previously registered address pool.

```go
ReleasePool(poolID string) error
```

For this API, the remote driver will receive a POST message to the URL `/IpamDriver.ReleasePool` with the following payload:

	{
		"PoolID": string
	}

Where:

* `PoolID` is the pool identifier

A successful response is empty:

    {}
    
### RequestAddress

This API is for reserving an ip address.

```go
RequestAddress(string, net.IP, map[string]string) (*net.IPNet, map[string]string, error)
```

For this API, the remote driver will receive a POST message to the URL `/IpamDriver.RequestAddress` with the following payload:

    {
		"PoolID":  string
		"Address": string
		"Options": map[string]string
    }
    
Where:

* `PoolID` is the pool identifier
* `Address` is the required address in regular IP form (A.B.C.D). If this address cannot be satisfied, the request fails. If empty, the IPAM driver chooses any available address on the pool
* `Options` are IPAM driver specific options


A successful response is in the form:


	{
		"Address": string
		"Data":    map[string]string
	}


Where:

* `Address` is the allocated address in CIDR format (A.B.C.D/MM)
* `Data` is some IPAM driver specific metadata

### ReleaseAddress

This API is for releasing an IP address.

For this API, the remote driver will receive a POST message to the URL `/IpamDriver.ReleaseAddress` with the following payload:

    {
		"PoolID": string
		"Address": string
    }
    
Where:

* `PoolID` is the pool identifier
* `Address` is the IP address to release



### GetCapabilities

During the driver registration, libnetwork will query the driver about its capabilities. It is not mandatory for the driver to support this URL endpoint. If driver does not support it, registration will succeed with empty capabilities automatically added to the internal driver handle.

During registration, the remote driver will receive a POST message to the URL `/IpamDriver.GetCapabilities` with no payload. The driver's response should have the form:


	{
		"RequiresMACAddress": bool
		"RequiresRequestReplay": bool
	}
	
	
	
## Capabilities

Capabilities are requirements, features the remote ipam driver can express during registration with libnetwork.
As of now libnetwork accepts the following capabilities:

### RequiresMACAddress

It is a boolean value which tells libnetwork whether the ipam driver needs to know the interface MAC address in order to properly process the `RequestAddress()` call.
If true, on `CreateEndpoint()` request, libnetwork will generate a random MAC address for the endpoint (if an explicit MAC address was not already provided by the user) and pass it to `RequestAddress()` when requesting the IP address inside the options map. The key will be the `netlabel.MacAddress` constant: `"com.docker.network.endpoint.macaddress"`.

### RequiresRequestReplay

It is a boolean value which tells libnetwork whether the ipam driver needs to receive the replay of the `RequestPool()` and `RequestAddress()` requests on daemon reload.  When libnetwork controller is initializing, it retrieves from local store the list of current local scope networks and, if this capability flag is set, it allows the IPAM driver to reconstruct the database of pools by replaying the `RequestPool()` requests for each pool and the `RequestAddress()` for each network gateway owned by the local networks. This can be useful to ipam drivers which decide not to persist the pools allocated to local scope networks.


## Appendix

A Go extension for the IPAM remote API is available at [docker/go-plugins-helpers/ipam](https://github.com/docker/go-plugins-helpers/tree/master/ipam)
