# Package: testing

## Purpose
Test helper package for constructing Endpoints API objects that pass validation, used in unit tests throughout Kubernetes.

## Key Types
- `Tweak` - Function type `func(*api.Endpoints)` for modifying Endpoints objects

## Key Functions
- `MakeEndpoints(name string, addrs []api.EndpointAddress, ports []api.EndpointPort, tweaks ...Tweak) *api.Endpoints` - Creates a valid Endpoints object with default namespace and the given addresses/ports
- `MakeEndpointAddress(ip string, pod string) api.EndpointAddress` - Creates an EndpointAddress with IP and pod TargetRef
- `MakeEndpointPort(name string, port int) api.EndpointPort` - Creates an EndpointPort with name and port number

## Design Pattern
Uses the functional options pattern (tweaks) to allow flexible customization of test objects while ensuring base validity for API validation.
