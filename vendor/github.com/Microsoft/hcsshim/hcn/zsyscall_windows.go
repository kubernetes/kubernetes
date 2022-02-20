// Code generated mksyscall_windows.exe DO NOT EDIT

package hcn

import (
	"syscall"
	"unsafe"

	"golang.org/x/sys/windows"
)

var _ unsafe.Pointer

// Do the interface allocations only once for common
// Errno values.
const (
	errnoERROR_IO_PENDING = 997
)

var (
	errERROR_IO_PENDING error = syscall.Errno(errnoERROR_IO_PENDING)
)

// errnoErr returns common boxed Errno values, to prevent
// allocations at runtime.
func errnoErr(e syscall.Errno) error {
	switch e {
	case 0:
		return nil
	case errnoERROR_IO_PENDING:
		return errERROR_IO_PENDING
	}
	// TODO: add more here, after collecting data on the common
	// error values see on Windows. (perhaps when running
	// all.bat?)
	return e
}

var (
	modiphlpapi       = windows.NewLazySystemDLL("iphlpapi.dll")
	modvmcompute      = windows.NewLazySystemDLL("vmcompute.dll")
	modcomputenetwork = windows.NewLazySystemDLL("computenetwork.dll")

	procSetCurrentThreadCompartmentId  = modiphlpapi.NewProc("SetCurrentThreadCompartmentId")
	procHNSCall                        = modvmcompute.NewProc("HNSCall")
	procHcnEnumerateNetworks           = modcomputenetwork.NewProc("HcnEnumerateNetworks")
	procHcnCreateNetwork               = modcomputenetwork.NewProc("HcnCreateNetwork")
	procHcnOpenNetwork                 = modcomputenetwork.NewProc("HcnOpenNetwork")
	procHcnModifyNetwork               = modcomputenetwork.NewProc("HcnModifyNetwork")
	procHcnQueryNetworkProperties      = modcomputenetwork.NewProc("HcnQueryNetworkProperties")
	procHcnDeleteNetwork               = modcomputenetwork.NewProc("HcnDeleteNetwork")
	procHcnCloseNetwork                = modcomputenetwork.NewProc("HcnCloseNetwork")
	procHcnEnumerateEndpoints          = modcomputenetwork.NewProc("HcnEnumerateEndpoints")
	procHcnCreateEndpoint              = modcomputenetwork.NewProc("HcnCreateEndpoint")
	procHcnOpenEndpoint                = modcomputenetwork.NewProc("HcnOpenEndpoint")
	procHcnModifyEndpoint              = modcomputenetwork.NewProc("HcnModifyEndpoint")
	procHcnQueryEndpointProperties     = modcomputenetwork.NewProc("HcnQueryEndpointProperties")
	procHcnDeleteEndpoint              = modcomputenetwork.NewProc("HcnDeleteEndpoint")
	procHcnCloseEndpoint               = modcomputenetwork.NewProc("HcnCloseEndpoint")
	procHcnEnumerateNamespaces         = modcomputenetwork.NewProc("HcnEnumerateNamespaces")
	procHcnCreateNamespace             = modcomputenetwork.NewProc("HcnCreateNamespace")
	procHcnOpenNamespace               = modcomputenetwork.NewProc("HcnOpenNamespace")
	procHcnModifyNamespace             = modcomputenetwork.NewProc("HcnModifyNamespace")
	procHcnQueryNamespaceProperties    = modcomputenetwork.NewProc("HcnQueryNamespaceProperties")
	procHcnDeleteNamespace             = modcomputenetwork.NewProc("HcnDeleteNamespace")
	procHcnCloseNamespace              = modcomputenetwork.NewProc("HcnCloseNamespace")
	procHcnEnumerateLoadBalancers      = modcomputenetwork.NewProc("HcnEnumerateLoadBalancers")
	procHcnCreateLoadBalancer          = modcomputenetwork.NewProc("HcnCreateLoadBalancer")
	procHcnOpenLoadBalancer            = modcomputenetwork.NewProc("HcnOpenLoadBalancer")
	procHcnModifyLoadBalancer          = modcomputenetwork.NewProc("HcnModifyLoadBalancer")
	procHcnQueryLoadBalancerProperties = modcomputenetwork.NewProc("HcnQueryLoadBalancerProperties")
	procHcnDeleteLoadBalancer          = modcomputenetwork.NewProc("HcnDeleteLoadBalancer")
	procHcnCloseLoadBalancer           = modcomputenetwork.NewProc("HcnCloseLoadBalancer")
	procHcnEnumerateSdnRoutes          = modcomputenetwork.NewProc("HcnEnumerateSdnRoutes")
	procHcnCreateSdnRoute              = modcomputenetwork.NewProc("HcnCreateSdnRoute")
	procHcnOpenSdnRoute                = modcomputenetwork.NewProc("HcnOpenSdnRoute")
	procHcnModifySdnRoute              = modcomputenetwork.NewProc("HcnModifySdnRoute")
	procHcnQuerySdnRouteProperties     = modcomputenetwork.NewProc("HcnQuerySdnRouteProperties")
	procHcnDeleteSdnRoute              = modcomputenetwork.NewProc("HcnDeleteSdnRoute")
	procHcnCloseSdnRoute               = modcomputenetwork.NewProc("HcnCloseSdnRoute")
)

func SetCurrentThreadCompartmentId(compartmentId uint32) (hr error) {
	r0, _, _ := syscall.Syscall(procSetCurrentThreadCompartmentId.Addr(), 1, uintptr(compartmentId), 0, 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func _hnsCall(method string, path string, object string, response **uint16) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(method)
	if hr != nil {
		return
	}
	var _p1 *uint16
	_p1, hr = syscall.UTF16PtrFromString(path)
	if hr != nil {
		return
	}
	var _p2 *uint16
	_p2, hr = syscall.UTF16PtrFromString(object)
	if hr != nil {
		return
	}
	return __hnsCall(_p0, _p1, _p2, response)
}

func __hnsCall(method *uint16, path *uint16, object *uint16, response **uint16) (hr error) {
	if hr = procHNSCall.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall6(procHNSCall.Addr(), 4, uintptr(unsafe.Pointer(method)), uintptr(unsafe.Pointer(path)), uintptr(unsafe.Pointer(object)), uintptr(unsafe.Pointer(response)), 0, 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcnEnumerateNetworks(query string, networks **uint16, result **uint16) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(query)
	if hr != nil {
		return
	}
	return _hcnEnumerateNetworks(_p0, networks, result)
}

func _hcnEnumerateNetworks(query *uint16, networks **uint16, result **uint16) (hr error) {
	if hr = procHcnEnumerateNetworks.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procHcnEnumerateNetworks.Addr(), 3, uintptr(unsafe.Pointer(query)), uintptr(unsafe.Pointer(networks)), uintptr(unsafe.Pointer(result)))
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcnCreateNetwork(id *_guid, settings string, network *hcnNetwork, result **uint16) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(settings)
	if hr != nil {
		return
	}
	return _hcnCreateNetwork(id, _p0, network, result)
}

func _hcnCreateNetwork(id *_guid, settings *uint16, network *hcnNetwork, result **uint16) (hr error) {
	if hr = procHcnCreateNetwork.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall6(procHcnCreateNetwork.Addr(), 4, uintptr(unsafe.Pointer(id)), uintptr(unsafe.Pointer(settings)), uintptr(unsafe.Pointer(network)), uintptr(unsafe.Pointer(result)), 0, 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcnOpenNetwork(id *_guid, network *hcnNetwork, result **uint16) (hr error) {
	if hr = procHcnOpenNetwork.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procHcnOpenNetwork.Addr(), 3, uintptr(unsafe.Pointer(id)), uintptr(unsafe.Pointer(network)), uintptr(unsafe.Pointer(result)))
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcnModifyNetwork(network hcnNetwork, settings string, result **uint16) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(settings)
	if hr != nil {
		return
	}
	return _hcnModifyNetwork(network, _p0, result)
}

func _hcnModifyNetwork(network hcnNetwork, settings *uint16, result **uint16) (hr error) {
	if hr = procHcnModifyNetwork.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procHcnModifyNetwork.Addr(), 3, uintptr(network), uintptr(unsafe.Pointer(settings)), uintptr(unsafe.Pointer(result)))
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcnQueryNetworkProperties(network hcnNetwork, query string, properties **uint16, result **uint16) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(query)
	if hr != nil {
		return
	}
	return _hcnQueryNetworkProperties(network, _p0, properties, result)
}

func _hcnQueryNetworkProperties(network hcnNetwork, query *uint16, properties **uint16, result **uint16) (hr error) {
	if hr = procHcnQueryNetworkProperties.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall6(procHcnQueryNetworkProperties.Addr(), 4, uintptr(network), uintptr(unsafe.Pointer(query)), uintptr(unsafe.Pointer(properties)), uintptr(unsafe.Pointer(result)), 0, 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcnDeleteNetwork(id *_guid, result **uint16) (hr error) {
	if hr = procHcnDeleteNetwork.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procHcnDeleteNetwork.Addr(), 2, uintptr(unsafe.Pointer(id)), uintptr(unsafe.Pointer(result)), 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcnCloseNetwork(network hcnNetwork) (hr error) {
	if hr = procHcnCloseNetwork.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procHcnCloseNetwork.Addr(), 1, uintptr(network), 0, 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcnEnumerateEndpoints(query string, endpoints **uint16, result **uint16) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(query)
	if hr != nil {
		return
	}
	return _hcnEnumerateEndpoints(_p0, endpoints, result)
}

func _hcnEnumerateEndpoints(query *uint16, endpoints **uint16, result **uint16) (hr error) {
	if hr = procHcnEnumerateEndpoints.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procHcnEnumerateEndpoints.Addr(), 3, uintptr(unsafe.Pointer(query)), uintptr(unsafe.Pointer(endpoints)), uintptr(unsafe.Pointer(result)))
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcnCreateEndpoint(network hcnNetwork, id *_guid, settings string, endpoint *hcnEndpoint, result **uint16) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(settings)
	if hr != nil {
		return
	}
	return _hcnCreateEndpoint(network, id, _p0, endpoint, result)
}

func _hcnCreateEndpoint(network hcnNetwork, id *_guid, settings *uint16, endpoint *hcnEndpoint, result **uint16) (hr error) {
	if hr = procHcnCreateEndpoint.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall6(procHcnCreateEndpoint.Addr(), 5, uintptr(network), uintptr(unsafe.Pointer(id)), uintptr(unsafe.Pointer(settings)), uintptr(unsafe.Pointer(endpoint)), uintptr(unsafe.Pointer(result)), 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcnOpenEndpoint(id *_guid, endpoint *hcnEndpoint, result **uint16) (hr error) {
	if hr = procHcnOpenEndpoint.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procHcnOpenEndpoint.Addr(), 3, uintptr(unsafe.Pointer(id)), uintptr(unsafe.Pointer(endpoint)), uintptr(unsafe.Pointer(result)))
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcnModifyEndpoint(endpoint hcnEndpoint, settings string, result **uint16) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(settings)
	if hr != nil {
		return
	}
	return _hcnModifyEndpoint(endpoint, _p0, result)
}

func _hcnModifyEndpoint(endpoint hcnEndpoint, settings *uint16, result **uint16) (hr error) {
	if hr = procHcnModifyEndpoint.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procHcnModifyEndpoint.Addr(), 3, uintptr(endpoint), uintptr(unsafe.Pointer(settings)), uintptr(unsafe.Pointer(result)))
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcnQueryEndpointProperties(endpoint hcnEndpoint, query string, properties **uint16, result **uint16) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(query)
	if hr != nil {
		return
	}
	return _hcnQueryEndpointProperties(endpoint, _p0, properties, result)
}

func _hcnQueryEndpointProperties(endpoint hcnEndpoint, query *uint16, properties **uint16, result **uint16) (hr error) {
	if hr = procHcnQueryEndpointProperties.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall6(procHcnQueryEndpointProperties.Addr(), 4, uintptr(endpoint), uintptr(unsafe.Pointer(query)), uintptr(unsafe.Pointer(properties)), uintptr(unsafe.Pointer(result)), 0, 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcnDeleteEndpoint(id *_guid, result **uint16) (hr error) {
	if hr = procHcnDeleteEndpoint.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procHcnDeleteEndpoint.Addr(), 2, uintptr(unsafe.Pointer(id)), uintptr(unsafe.Pointer(result)), 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcnCloseEndpoint(endpoint hcnEndpoint) (hr error) {
	if hr = procHcnCloseEndpoint.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procHcnCloseEndpoint.Addr(), 1, uintptr(endpoint), 0, 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcnEnumerateNamespaces(query string, namespaces **uint16, result **uint16) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(query)
	if hr != nil {
		return
	}
	return _hcnEnumerateNamespaces(_p0, namespaces, result)
}

func _hcnEnumerateNamespaces(query *uint16, namespaces **uint16, result **uint16) (hr error) {
	if hr = procHcnEnumerateNamespaces.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procHcnEnumerateNamespaces.Addr(), 3, uintptr(unsafe.Pointer(query)), uintptr(unsafe.Pointer(namespaces)), uintptr(unsafe.Pointer(result)))
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcnCreateNamespace(id *_guid, settings string, namespace *hcnNamespace, result **uint16) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(settings)
	if hr != nil {
		return
	}
	return _hcnCreateNamespace(id, _p0, namespace, result)
}

func _hcnCreateNamespace(id *_guid, settings *uint16, namespace *hcnNamespace, result **uint16) (hr error) {
	if hr = procHcnCreateNamespace.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall6(procHcnCreateNamespace.Addr(), 4, uintptr(unsafe.Pointer(id)), uintptr(unsafe.Pointer(settings)), uintptr(unsafe.Pointer(namespace)), uintptr(unsafe.Pointer(result)), 0, 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcnOpenNamespace(id *_guid, namespace *hcnNamespace, result **uint16) (hr error) {
	if hr = procHcnOpenNamespace.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procHcnOpenNamespace.Addr(), 3, uintptr(unsafe.Pointer(id)), uintptr(unsafe.Pointer(namespace)), uintptr(unsafe.Pointer(result)))
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcnModifyNamespace(namespace hcnNamespace, settings string, result **uint16) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(settings)
	if hr != nil {
		return
	}
	return _hcnModifyNamespace(namespace, _p0, result)
}

func _hcnModifyNamespace(namespace hcnNamespace, settings *uint16, result **uint16) (hr error) {
	if hr = procHcnModifyNamespace.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procHcnModifyNamespace.Addr(), 3, uintptr(namespace), uintptr(unsafe.Pointer(settings)), uintptr(unsafe.Pointer(result)))
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcnQueryNamespaceProperties(namespace hcnNamespace, query string, properties **uint16, result **uint16) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(query)
	if hr != nil {
		return
	}
	return _hcnQueryNamespaceProperties(namespace, _p0, properties, result)
}

func _hcnQueryNamespaceProperties(namespace hcnNamespace, query *uint16, properties **uint16, result **uint16) (hr error) {
	if hr = procHcnQueryNamespaceProperties.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall6(procHcnQueryNamespaceProperties.Addr(), 4, uintptr(namespace), uintptr(unsafe.Pointer(query)), uintptr(unsafe.Pointer(properties)), uintptr(unsafe.Pointer(result)), 0, 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcnDeleteNamespace(id *_guid, result **uint16) (hr error) {
	if hr = procHcnDeleteNamespace.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procHcnDeleteNamespace.Addr(), 2, uintptr(unsafe.Pointer(id)), uintptr(unsafe.Pointer(result)), 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcnCloseNamespace(namespace hcnNamespace) (hr error) {
	if hr = procHcnCloseNamespace.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procHcnCloseNamespace.Addr(), 1, uintptr(namespace), 0, 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcnEnumerateLoadBalancers(query string, loadBalancers **uint16, result **uint16) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(query)
	if hr != nil {
		return
	}
	return _hcnEnumerateLoadBalancers(_p0, loadBalancers, result)
}

func _hcnEnumerateLoadBalancers(query *uint16, loadBalancers **uint16, result **uint16) (hr error) {
	if hr = procHcnEnumerateLoadBalancers.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procHcnEnumerateLoadBalancers.Addr(), 3, uintptr(unsafe.Pointer(query)), uintptr(unsafe.Pointer(loadBalancers)), uintptr(unsafe.Pointer(result)))
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcnCreateLoadBalancer(id *_guid, settings string, loadBalancer *hcnLoadBalancer, result **uint16) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(settings)
	if hr != nil {
		return
	}
	return _hcnCreateLoadBalancer(id, _p0, loadBalancer, result)
}

func _hcnCreateLoadBalancer(id *_guid, settings *uint16, loadBalancer *hcnLoadBalancer, result **uint16) (hr error) {
	if hr = procHcnCreateLoadBalancer.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall6(procHcnCreateLoadBalancer.Addr(), 4, uintptr(unsafe.Pointer(id)), uintptr(unsafe.Pointer(settings)), uintptr(unsafe.Pointer(loadBalancer)), uintptr(unsafe.Pointer(result)), 0, 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcnOpenLoadBalancer(id *_guid, loadBalancer *hcnLoadBalancer, result **uint16) (hr error) {
	if hr = procHcnOpenLoadBalancer.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procHcnOpenLoadBalancer.Addr(), 3, uintptr(unsafe.Pointer(id)), uintptr(unsafe.Pointer(loadBalancer)), uintptr(unsafe.Pointer(result)))
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcnModifyLoadBalancer(loadBalancer hcnLoadBalancer, settings string, result **uint16) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(settings)
	if hr != nil {
		return
	}
	return _hcnModifyLoadBalancer(loadBalancer, _p0, result)
}

func _hcnModifyLoadBalancer(loadBalancer hcnLoadBalancer, settings *uint16, result **uint16) (hr error) {
	if hr = procHcnModifyLoadBalancer.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procHcnModifyLoadBalancer.Addr(), 3, uintptr(loadBalancer), uintptr(unsafe.Pointer(settings)), uintptr(unsafe.Pointer(result)))
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcnQueryLoadBalancerProperties(loadBalancer hcnLoadBalancer, query string, properties **uint16, result **uint16) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(query)
	if hr != nil {
		return
	}
	return _hcnQueryLoadBalancerProperties(loadBalancer, _p0, properties, result)
}

func _hcnQueryLoadBalancerProperties(loadBalancer hcnLoadBalancer, query *uint16, properties **uint16, result **uint16) (hr error) {
	if hr = procHcnQueryLoadBalancerProperties.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall6(procHcnQueryLoadBalancerProperties.Addr(), 4, uintptr(loadBalancer), uintptr(unsafe.Pointer(query)), uintptr(unsafe.Pointer(properties)), uintptr(unsafe.Pointer(result)), 0, 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcnDeleteLoadBalancer(id *_guid, result **uint16) (hr error) {
	if hr = procHcnDeleteLoadBalancer.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procHcnDeleteLoadBalancer.Addr(), 2, uintptr(unsafe.Pointer(id)), uintptr(unsafe.Pointer(result)), 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcnCloseLoadBalancer(loadBalancer hcnLoadBalancer) (hr error) {
	if hr = procHcnCloseLoadBalancer.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procHcnCloseLoadBalancer.Addr(), 1, uintptr(loadBalancer), 0, 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcnEnumerateRoutes(query string, routes **uint16, result **uint16) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(query)
	if hr != nil {
		return
	}
	return _hcnEnumerateRoutes(_p0, routes, result)
}

func _hcnEnumerateRoutes(query *uint16, routes **uint16, result **uint16) (hr error) {
	if hr = procHcnEnumerateSdnRoutes.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procHcnEnumerateSdnRoutes.Addr(), 3, uintptr(unsafe.Pointer(query)), uintptr(unsafe.Pointer(routes)), uintptr(unsafe.Pointer(result)))
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcnCreateRoute(id *_guid, settings string, route *hcnRoute, result **uint16) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(settings)
	if hr != nil {
		return
	}
	return _hcnCreateRoute(id, _p0, route, result)
}

func _hcnCreateRoute(id *_guid, settings *uint16, route *hcnRoute, result **uint16) (hr error) {
	if hr = procHcnCreateSdnRoute.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall6(procHcnCreateSdnRoute.Addr(), 4, uintptr(unsafe.Pointer(id)), uintptr(unsafe.Pointer(settings)), uintptr(unsafe.Pointer(route)), uintptr(unsafe.Pointer(result)), 0, 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcnOpenRoute(id *_guid, route *hcnRoute, result **uint16) (hr error) {
	if hr = procHcnOpenSdnRoute.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procHcnOpenSdnRoute.Addr(), 3, uintptr(unsafe.Pointer(id)), uintptr(unsafe.Pointer(route)), uintptr(unsafe.Pointer(result)))
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcnModifyRoute(route hcnRoute, settings string, result **uint16) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(settings)
	if hr != nil {
		return
	}
	return _hcnModifyRoute(route, _p0, result)
}

func _hcnModifyRoute(route hcnRoute, settings *uint16, result **uint16) (hr error) {
	if hr = procHcnModifySdnRoute.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procHcnModifySdnRoute.Addr(), 3, uintptr(route), uintptr(unsafe.Pointer(settings)), uintptr(unsafe.Pointer(result)))
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcnQueryRouteProperties(route hcnRoute, query string, properties **uint16, result **uint16) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(query)
	if hr != nil {
		return
	}
	return _hcnQueryRouteProperties(route, _p0, properties, result)
}

func _hcnQueryRouteProperties(route hcnRoute, query *uint16, properties **uint16, result **uint16) (hr error) {
	if hr = procHcnQuerySdnRouteProperties.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall6(procHcnQuerySdnRouteProperties.Addr(), 4, uintptr(route), uintptr(unsafe.Pointer(query)), uintptr(unsafe.Pointer(properties)), uintptr(unsafe.Pointer(result)), 0, 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcnDeleteRoute(id *_guid, result **uint16) (hr error) {
	if hr = procHcnDeleteSdnRoute.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procHcnDeleteSdnRoute.Addr(), 2, uintptr(unsafe.Pointer(id)), uintptr(unsafe.Pointer(result)), 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcnCloseRoute(route hcnRoute) (hr error) {
	if hr = procHcnCloseSdnRoute.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procHcnCloseSdnRoute.Addr(), 1, uintptr(route), 0, 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}
