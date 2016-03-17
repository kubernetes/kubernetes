# Gophercloud: an OpenStack SDK for Go
[![Build Status](https://travis-ci.org/rackspace/gophercloud.svg?branch=master)](https://travis-ci.org/rackspace/gophercloud) [![Coverage Status](https://coveralls.io/repos/rackspace/gophercloud/badge.png)](https://coveralls.io/r/rackspace/gophercloud)

Gophercloud is a flexible SDK that allows you to consume and work with OpenStack
clouds in a simple and idiomatic way using golang. Many services are supported,
including Compute, Block Storage, Object Storage, Networking, and Identity.
Each service API is backed with getting started guides, code samples, reference
documentation, unit tests and acceptance tests.

## Useful links

* [Gophercloud homepage](http://gophercloud.io)
* [Reference documentation](http://godoc.org/github.com/rackspace/gophercloud)
* [Getting started guides](http://gophercloud.io/docs)
* [Effective Go](https://golang.org/doc/effective_go.html)

## How to install

Before installing, you need to ensure that your [GOPATH environment variable](https://golang.org/doc/code.html#GOPATH)
is pointing to an appropriate directory where you want to install Gophercloud:

```bash
mkdir $HOME/go
export GOPATH=$HOME/go
```

To protect yourself against changes in your dependencies, we highly recommend choosing a
[dependency management solution](https://github.com/golang/go/wiki/PackageManagementTools) for
your projects, such as [godep](https://github.com/tools/godep). Once this is set up, you can install
Gophercloud as a dependency like so:

```bash
go get github.com/rackspace/gophercloud

# Edit your code to import relevant packages from "github.com/rackspace/gophercloud"

godep save ./...
```

This will install all the source files you need into a `Godeps/_workspace` directory, which is
referenceable from your own source files when you use the `godep go` command.

## Getting started

### Credentials

Because you'll be hitting an API, you will need to retrieve your OpenStack
credentials and either store them as environment variables or in your local Go
files. The first method is recommended because it decouples credential
information from source code, allowing you to push the latter to your version
control system without any security risk.

You will need to retrieve the following:

* username
* password
* tenant name or tenant ID
* a valid Keystone identity URL

For users that have the OpenStack dashboard installed, there's a shortcut. If
you visit the `project/access_and_security` path in Horizon and click on the
"Download OpenStack RC File" button at the top right hand corner, you will
download a bash file that exports all of your access details to environment
variables. To execute the file, run `source admin-openrc.sh` and you will be
prompted for your password.

### Authentication

Once you have access to your credentials, you can begin plugging them into
Gophercloud. The next step is authentication, and this is handled by a base
"Provider" struct. To get one, you can either pass in your credentials
explicitly, or tell Gophercloud to use environment variables:

```go
import (
  "github.com/rackspace/gophercloud"
  "github.com/rackspace/gophercloud/openstack"
  "github.com/rackspace/gophercloud/openstack/utils"
)

// Option 1: Pass in the values yourself
opts := gophercloud.AuthOptions{
  IdentityEndpoint: "https://my-openstack.com:5000/v2.0",
  Username: "{username}",
  Password: "{password}",
  TenantID: "{tenant_id}",
}

// Option 2: Use a utility function to retrieve all your environment variables
opts, err := openstack.AuthOptionsFromEnv()
```

Once you have the `opts` variable, you can pass it in and get back a
`ProviderClient` struct:

```go
provider, err := openstack.AuthenticatedClient(opts)
```

The `ProviderClient` is the top-level client that all of your OpenStack services
derive from. The provider contains all of the authentication details that allow
your Go code to access the API - such as the base URL and token ID.

### Provision a server

Once we have a base Provider, we inject it as a dependency into each OpenStack
service. In order to work with the Compute API, we need a Compute service
client; which can be created like so:

```go
client, err := openstack.NewComputeV2(provider, gophercloud.EndpointOpts{
  Region: os.Getenv("OS_REGION_NAME"),
})
```

We then use this `client` for any Compute API operation we want. In our case,
we want to provision a new server - so we invoke the `Create` method and pass
in the flavor ID (hardware specification) and image ID (operating system) we're
interested in:

```go
import "github.com/rackspace/gophercloud/openstack/compute/v2/servers"

server, err := servers.Create(client, servers.CreateOpts{
  Name:      "My new server!",
  FlavorRef: "flavor_id",
  ImageRef:  "image_id",
}).Extract()
```

If you are unsure about what images and flavors are, you can read our [Compute
Getting Started guide](http://gophercloud.io/docs/compute). The above code
sample creates a new server with the parameters, and embodies the new resource
in the `server` variable (a
[`servers.Server`](http://godoc.org/github.com/rackspace/gophercloud) struct).

### Next steps

Cool! You've handled authentication, got your `ProviderClient` and provisioned
a new server. You're now ready to use more OpenStack services.

* [Getting started with Compute](http://gophercloud.io/docs/compute)
* [Getting started with Object Storage](http://gophercloud.io/docs/object-storage)
* [Getting started with Networking](http://gophercloud.io/docs/networking)
* [Getting started with Block Storage](http://gophercloud.io/docs/block-storage)
* [Getting started with Identity](http://gophercloud.io/docs/identity)

## Contributing

Engaging the community and lowering barriers for contributors is something we
care a lot about. For this reason, we've taken the time to write a [contributing
guide](./CONTRIBUTING.md) for folks interested in getting involved in our project.
If you're not sure how you can get involved, feel free to submit an issue or
[contact us](https://developer.rackspace.com/support/). You don't need to be a
Go expert - all members of the community are welcome!

## Help and feedback

If you're struggling with something or have spotted a potential bug, feel free
to submit an issue to our [bug tracker](/issues) or [contact us directly](https://developer.rackspace.com/support/).
