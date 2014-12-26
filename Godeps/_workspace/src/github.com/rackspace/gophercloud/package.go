/*
Package gophercloud provides a multi-vendor interface to OpenStack-compatible
clouds. The library has a three-level hierarchy: providers, services, and
resources.

Provider structs represent the service providers that offer and manage a
collection of services. Examples of providers include: OpenStack, Rackspace,
HP. These are defined like so:

  opts := gophercloud.AuthOptions{
    IdentityEndpoint: "https://my-openstack.com:5000/v2.0",
    Username: "{username}",
    Password: "{password}",
    TenantID: "{tenant_id}",
  }

  provider, err := openstack.AuthenticatedClient(opts)

Service structs are specific to a provider and handle all of the logic and
operations for a particular OpenStack service. Examples of services include:
Compute, Object Storage, Block Storage. In order to define one, you need to
pass in the parent provider, like so:

  opts := gophercloud.EndpointOpts{Region: "RegionOne"}

  client := openstack.NewComputeV2(provider, opts)

Resource structs are the domain models that services make use of in order
to work with and represent the state of API resources:

  server, err := servers.Get(client, "{serverId}").Extract()

Another convention is to return Result structs for API operations, which allow
you to access the HTTP headers, response body, and associated errors with the
network transaction. To get a resource struct, you then call the Extract
method which is chained to the response.
*/
package gophercloud
