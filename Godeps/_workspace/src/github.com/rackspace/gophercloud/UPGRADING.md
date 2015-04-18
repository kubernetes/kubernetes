# Upgrading to v1.0.0

With the arrival of this new major version increment, the unfortunate news is
that breaking changes have been introduced to existing services. The API
has been completely rewritten from the ground up to make the library more
extensible, maintainable and easy-to-use.

Below we've compiled upgrade instructions for the various services that
existed before. If you have a specific issue that is not addressed below,
please [submit an issue](/issues/new) or
[e-mail our support team](https://developer.rackspace.com/support/).

* [Authentication](#authentication)
* [Servers](#servers)
  * [List servers](#list-servers)
  * [Get server details](#get-server-details)
  * [Create server](#create-server)
  * [Resize server](#resize-server)
  * [Reboot server](#reboot-server)
  * [Update server](#update-server)
  * [Rebuild server](#rebuild-server)
  * [Change admin password](#change-admin-password)
  * [Delete server](#delete-server)
  * [Rescue server](#rescue-server)
* [Images and flavors](#images-and-flavors)
  * [List images](#list-images)
  * [List flavors](#list-flavors)
  * [Create/delete image](#createdelete-image)
* [Other](#other)
  * [List keypairs](#list-keypairs)
  * [Create/delete keypair](#createdelete-keypair)
  * [List IP addresses](#list-ip-addresses)

# Authentication

One of the major differences that this release introduces is the level of
sub-packaging to differentiate between services and providers. You now have
the option of authenticating with OpenStack and other providers (like Rackspace).

To authenticate with a vanilla OpenStack installation, you can either specify
your credentials like this:

```go
import (
  "github.com/rackspace/gophercloud"
  "github.com/rackspace/gophercloud/openstack"
)

opts := gophercloud.AuthOptions{
  IdentityEndpoint: "https://my-openstack.com:5000/v2.0",
  Username: "{username}",
  Password: "{password}",
  TenantID: "{tenant_id}",
}
```

Or have them pulled in through environment variables, like this:

```go
opts, err := openstack.AuthOptionsFromEnv()
```

Once you have your `AuthOptions` struct, you pass it in to get back a `Provider`,
like so:

```go
provider, err := openstack.AuthenticatedClient(opts)
```

This provider is the top-level structure that all services are created from.

# Servers

Before you can interact with the Compute API, you need to retrieve a
`gophercloud.ServiceClient`. To do this:

```go
// Define your region, etc.
opts := gophercloud.EndpointOpts{Region: "RegionOne"}

client, err := openstack.NewComputeV2(provider, opts)
```

## List servers

All operations that involve API collections (servers, flavors, images) now use
the `pagination.Pager` interface. This interface represents paginated entities
that can be iterated over.

Once you have a Pager, you can then pass a callback function into its `EachPage`
method, and this will allow you to traverse over the collection and execute
arbitrary functionality. So, an example with list servers:

```go
import (
  "fmt"
  "github.com/rackspace/gophercloud/pagination"
  "github.com/rackspace/gophercloud/openstack/compute/v2/servers"
)

// We have the option of filtering the server list. If we want the full
// collection, leave it as an empty struct or nil
opts := servers.ListOpts{Name: "server_1"}

// Retrieve a pager (i.e. a paginated collection)
pager := servers.List(client, opts)

// Define an anonymous function to be executed on each page's iteration
err := pager.EachPage(func(page pagination.Page) (bool, error) {
  serverList, err := servers.ExtractServers(page)

  // `s' will be a servers.Server struct
  for _, s := range serverList {
    fmt.Printf("We have a server. ID=%s, Name=%s", s.ID, s.Name)
  }
})
```

## Get server details

```go
import "github.com/rackspace/gophercloud/openstack/compute/v2/servers"

// Get the HTTP result
response := servers.Get(client, "server_id")

// Extract a Server struct from the response
server, err := response.Extract()
```

## Create server

```go
import "github.com/rackspace/gophercloud/openstack/compute/v2/servers"

// Define our options
opts := servers.CreateOpts{
  Name: "new_server",
  FlavorRef: "flavorID",
  ImageRef: "imageID",
}

// Get our response
response := servers.Create(client, opts)

// Extract
server, err := response.Extract()
```

## Change admin password

```go
import "github.com/rackspace/gophercloud/openstack/compute/v2/servers"

result := servers.ChangeAdminPassword(client, "server_id", "newPassword_&123")
```

## Resize server

```go
import "github.com/rackspace/gophercloud/openstack/compute/v2/servers"

result := servers.Resize(client, "server_id", "new_flavor_id")
```

## Reboot server

```go
import "github.com/rackspace/gophercloud/openstack/compute/v2/servers"

// You have a choice of two reboot methods: servers.SoftReboot or servers.HardReboot
result := servers.Reboot(client, "server_id", servers.SoftReboot)
```

## Update server

```go
import "github.com/rackspace/gophercloud/openstack/compute/v2/servers"

opts := servers.UpdateOpts{Name: "new_name"}

server, err := servers.Update(client, "server_id", opts).Extract()
```

## Rebuild server

```go
import "github.com/rackspace/gophercloud/openstack/compute/v2/servers"

// You have the option of specifying additional options
opts := RebuildOpts{
  Name:      "new_name",
  AdminPass: "admin_password",
  ImageID:   "image_id",
  Metadata:  map[string]string{"owner": "me"},
}

result := servers.Rebuild(client, "server_id", opts)

// You can extract a servers.Server struct from the HTTP response
server, err := result.Extract()
```

## Delete server

```go
import "github.com/rackspace/gophercloud/openstack/compute/v2/servers"

response := servers.Delete(client, "server_id")
```

## Rescue server

The server rescue extension for Compute is not currently supported.

# Images and flavors

## List images

As with listing servers (see above), you first retrieve a Pager, and then pass
in a callback over each page:

```go
import (
  "github.com/rackspace/gophercloud/pagination"
  "github.com/rackspace/gophercloud/openstack/compute/v2/images"
)

// We have the option of filtering the image list. If we want the full
// collection, leave it as an empty struct
opts := images.ListOpts{ChangesSince: "2014-01-01T01:02:03Z", Name: "Ubuntu 12.04"}

// Retrieve a pager (i.e. a paginated collection)
pager := images.List(client, opts)

// Define an anonymous function to be executed on each page's iteration
err := pager.EachPage(func(page pagination.Page) (bool, error) {
  imageList, err := images.ExtractImages(page)

  for _, i := range imageList {
    // "i" will be an images.Image
  }
})
```

## List flavors

```go
import (
  "github.com/rackspace/gophercloud/pagination"
  "github.com/rackspace/gophercloud/openstack/compute/v2/flavors"
)

// We have the option of filtering the flavor list. If we want the full
// collection, leave it as an empty struct
opts := flavors.ListOpts{ChangesSince: "2014-01-01T01:02:03Z", MinRAM: 4}

// Retrieve a pager (i.e. a paginated collection)
pager := flavors.List(client, opts)

// Define an anonymous function to be executed on each page's iteration
err := pager.EachPage(func(page pagination.Page) (bool, error) {
  flavorList, err := networks.ExtractFlavors(page)

  for _, f := range flavorList {
    // "f" will be a flavors.Flavor
  }
})
```

## Create/delete image

Image management has been shifted to Glance, but unfortunately this service is
not supported as of yet. You can, however, list Compute images like so:

```go
import "github.com/rackspace/gophercloud/openstack/compute/v2/images"

// Retrieve a pager (i.e. a paginated collection)
pager := images.List(client, opts)

// Define an anonymous function to be executed on each page's iteration
err := pager.EachPage(func(page pagination.Page) (bool, error) {
  imageList, err := images.ExtractImages(page)

  for _, i := range imageList {
    // "i" will be an images.Image
  }
})
```

# Other

## List keypairs

```go
import "github.com/rackspace/gophercloud/openstack/compute/v2/extensions/keypairs"

// Retrieve a pager (i.e. a paginated collection)
pager := keypairs.List(client, opts)

// Define an anonymous function to be executed on each page's iteration
err := pager.EachPage(func(page pagination.Page) (bool, error) {
  keyList, err := keypairs.ExtractKeyPairs(page)

  for _, k := range keyList {
    // "k" will be a keypairs.KeyPair
  }
})
```

## Create/delete keypairs

To create a new keypair, you need to specify its name and, optionally, a
pregenerated OpenSSH-formatted public key.

```go
import "github.com/rackspace/gophercloud/openstack/compute/v2/extensions/keypairs"

opts := keypairs.CreateOpts{
  Name: "new_key",
  PublicKey: "...",
}

response := keypairs.Create(client, opts)

key, err := response.Extract()
```

To delete an existing keypair:

```go
response := keypairs.Delete(client, "keypair_id")
```

## List IP addresses

This operation is not currently supported.
