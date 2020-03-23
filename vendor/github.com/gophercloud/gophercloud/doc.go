/*
Package gophercloud provides a multi-vendor interface to OpenStack-compatible
clouds. The library has a three-level hierarchy: providers, services, and
resources.

Authenticating with Providers

Provider structs represent the cloud providers that offer and manage a
collection of services. You will generally want to create one Provider
client per OpenStack cloud.

	It is now recommended to use the `clientconfig` package found at
	https://github.com/gophercloud/utils/tree/master/openstack/clientconfig
	for all authentication purposes.

	The below documentation is still relevant. clientconfig simply implements
	the below and presents it in an easier and more flexible way.

Use your OpenStack credentials to create a Provider client.  The
IdentityEndpoint is typically refered to as "auth_url" or "OS_AUTH_URL" in
information provided by the cloud operator. Additionally, the cloud may refer to
TenantID or TenantName as project_id and project_name. Credentials are
specified like so:

	opts := gophercloud.AuthOptions{
		IdentityEndpoint: "https://openstack.example.com:5000/v2.0",
		Username: "{username}",
		Password: "{password}",
		TenantID: "{tenant_id}",
	}

	provider, err := openstack.AuthenticatedClient(opts)

You can authenticate with a token by doing:

	opts := gophercloud.AuthOptions{
		IdentityEndpoint: "https://openstack.example.com:5000/v2.0",
		TokenID:  "{token_id}",
		TenantID: "{tenant_id}",
	}

	provider, err := openstack.AuthenticatedClient(opts)

You may also use the openstack.AuthOptionsFromEnv() helper function. This
function reads in standard environment variables frequently found in an
OpenStack `openrc` file. Again note that Gophercloud currently uses "tenant"
instead of "project".

	opts, err := openstack.AuthOptionsFromEnv()
	provider, err := openstack.AuthenticatedClient(opts)

Service Clients

Service structs are specific to a provider and handle all of the logic and
operations for a particular OpenStack service. Examples of services include:
Compute, Object Storage, Block Storage. In order to define one, you need to
pass in the parent provider, like so:

	opts := gophercloud.EndpointOpts{Region: "RegionOne"}

	client, err := openstack.NewComputeV2(provider, opts)

Resources

Resource structs are the domain models that services make use of in order
to work with and represent the state of API resources:

	server, err := servers.Get(client, "{serverId}").Extract()

Intermediate Result structs are returned for API operations, which allow
generic access to the HTTP headers, response body, and any errors associated
with the network transaction. To turn a result into a usable resource struct,
you must call the Extract method which is chained to the response, or an
Extract function from an applicable extension:

	result := servers.Get(client, "{serverId}")

	// Attempt to extract the disk configuration from the OS-DCF disk config
	// extension:
	config, err := diskconfig.ExtractGet(result)

All requests that enumerate a collection return a Pager struct that is used to
iterate through the results one page at a time. Use the EachPage method on that
Pager to handle each successive Page in a closure, then use the appropriate
extraction method from that request's package to interpret that Page as a slice
of results:

	err := servers.List(client, nil).EachPage(func (page pagination.Page) (bool, error) {
		s, err := servers.ExtractServers(page)
		if err != nil {
			return false, err
		}

		// Handle the []servers.Server slice.

		// Return "false" or an error to prematurely stop fetching new pages.
		return true, nil
	})

If you want to obtain the entire collection of pages without doing any
intermediary processing on each page, you can use the AllPages method:

	allPages, err := servers.List(client, nil).AllPages()
	allServers, err := servers.ExtractServers(allPages)

This top-level package contains utility functions and data types that are used
throughout the provider and service packages. Of particular note for end users
are the AuthOptions and EndpointOpts structs.
*/
package gophercloud
