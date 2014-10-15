package gophercloud

import (
	"fmt"
)

// ErrNotImplemented should be used only while developing new SDK features.
// No established function or method will ever produce this error.
var ErrNotImplemented = fmt.Errorf("Not implemented")

// ErrProvider errors occur when attempting to reference an unsupported
// provider.  More often than not, this error happens due to a typo in
// the name.
var ErrProvider = fmt.Errorf("Missing or incorrect provider")

// ErrCredentials errors happen when attempting to authenticate using a
// set of credentials not recognized by the Authenticate() method.
// For example, not providing a username or password when attempting to
// authenticate against an Identity V2 API.
var ErrCredentials = fmt.Errorf("Missing or incomplete credentials")

// ErrConfiguration errors happen when attempting to add a new provider, and
// the provider added lacks a correct or consistent configuration.
// For example, all providers must expose at least an Identity V2 API
// for authentication; if this endpoint isn't specified, you may receive
// this error when attempting to register it against a context.
var ErrConfiguration = fmt.Errorf("Missing or incomplete configuration")

// ErrError errors happen when you attempt to discover the response code
// responsible for a previous request bombing with an error, but pass in an
// error interface which doesn't belong to the web client.
var ErrError = fmt.Errorf("Attempt to solicit actual HTTP response code from error entity which doesn't know")

// WarnUnauthoritative warnings happen when a service believes its response
// to be correct, but is not in a position of knowing for sure at the moment.
// For example, the service could be responding with cached data that has
// exceeded its time-to-live setting, but which has not yet received an official
// update from an authoritative source.
var WarnUnauthoritative = fmt.Errorf("Unauthoritative data")
