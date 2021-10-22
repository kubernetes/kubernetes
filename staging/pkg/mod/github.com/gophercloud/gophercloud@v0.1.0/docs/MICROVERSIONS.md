# Microversions

## Table of Contents

* [Introduction](#introduction)
* [Client Configuration](#client-configuration)
* [Gophercloud Developer Information](#gophercloud-developer-information)
* [Application Developer Information](#application-developer-information)

## Introduction

Microversions are an OpenStack API ability which allows developers to add and
remove features while still retaining backwards compatibility for all prior
versions of the API.

More information can be found here:

> Note: these links are not an exhaustive reference for microversions.

* http://specs.openstack.org/openstack/api-wg/guidelines/microversion_specification.html
* https://developer.openstack.org/api-guide/compute/microversions.html
* https://github.com/openstack/keystoneauth/blob/master/doc/source/using-sessions.rst

## Client Configuration

You can set a specific microversion on a Service Client by doing the following:

```go
client, err := openstack.NewComputeV2(providerClient, nil)
client.Microversion = "2.52"
```

## Gophercloud Developer Information

Microversions change several aspects about API interaction.

### Existing Fields, New Values

This is when an existing field behaves like an "enum" and a new valid value
is possible by setting the client's microversion to a specific version.

An example of this can be seen with Nova/Compute's Server Group `policy` field
and the introduction of the [`soft-affinity`](https://developer.openstack.org/api-ref/compute/?expanded=create-server-group-detail#create-server-group)
value.

Unless Gophercloud is limiting the valid values that are passed to the
Nova/Compute service, no changes are required in Gophercloud.

### New Request Fields

This is when a microversion enables a new field to be used in an API request.
When implementing this kind of change, it is imperative that the field has
the `omitempty` attribute set. If `omitempty` is not set, then the field will
be used for _all_ microversions and possibly cause an error from the API
service. You may need to use a pointer field in order for this to work.

When adding a new field, please make sure to include a GoDoc comment about
what microversions the field is valid for.

### New Response Fields

This is when a microversion includes new fields in the API response. The
correct way of implementing this in Gophercloud is to _not_ add the field
to the resource's "result" struct (in the `results.go` file), but instead
add a custom "extract" method to a new `microversions.go` file. This is
to ensure that base API interaction does not break with the introduction
of new fields.

### Modified Response Fields

This is when a microversion modifies an existing field in an API response
to be formatted differently than the base API. Research is still ongoing
on how to best handle this scenario in Gophercloud.

## Application Developer Information

Gophercloud does not perform any validation checks on the API request to make
sure it is valid for a specific microversion. It is up to you to ensure that
the API request is using the correct fields and functions for the microversion.
