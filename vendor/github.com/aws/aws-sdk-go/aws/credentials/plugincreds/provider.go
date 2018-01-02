// +build go1.8

// Package plugincreds implements a credentials provider sourced from a Go
// plugin. This package allows you to use a Go plugin to retrieve AWS credentials
// for the SDK to use for service API calls.
//
// As of Go 1.8 plugins are only supported on the Linux platform.
//
// Plugin Symbol Name
//
// The "GetAWSSDKCredentialProvider" is the symbol name that will be used to
// lookup the credentials provider getter from the plugin. If you want to use a
// custom symbol name you should use GetPluginProviderFnsByName to lookup the
// symbol by a custom name.
//
// This symbol is a function that returns two additional functions. One to
// retrieve the credentials, and another to determine if the credentials have
// expired.
//
// Plugin Symbol Signature
//
// The plugin credential provider requires the symbol to match the
// following signature.
//
//   func() (RetrieveFn func() (key, secret, token string, err error), IsExpiredFn func() bool)
//
// Plugin Implementation Exmaple
//
// The following is an example implementation of a SDK credential provider using
// the plugin provider in this package. See the SDK's example/aws/credential/plugincreds/plugin
// folder for a runnable example of this.
//
//   package main
//
//   func main() {}
//
//   var myCredProvider provider
//
//   // Build: go build -o plugin.so -buildmode=plugin plugin.go
//   func init() {
//   	// Initialize a mock credential provider with stubs
//   	myCredProvider = provider{"a","b","c"}
//   }
//
//   // GetAWSSDKCredentialProvider is the symbol SDK will lookup and use to
//   // get the credential provider's retrieve and isExpired functions.
//   func GetAWSSDKCredentialProvider() (func() (key, secret, token string, err error), func() bool) {
//   	return myCredProvider.Retrieve,	myCredProvider.IsExpired
//   }
//
//   // mock implementation of a type that returns retrieves credentials and
//   // returns if they have expired.
//   type provider struct {
//   	key, secret, token string
//   }
//
//   func (p provider) Retrieve() (key, secret, token string, err error) {
//   	return p.key, p.secret, p.token, nil
//   }
//
//   func (p *provider) IsExpired() bool {
//   	return false;
//   }
//
// Configuring SDK for Plugin Credentials
//
// To configure the SDK to use a plugin's credential provider you'll need to first
// open the plugin file using the plugin standard library package. Once you have
// a handle to the plugin you can use the NewCredentials function of this package
// to create a new credentials.Credentials value that can be set as the
// credentials loader of a Session or Config. See the SDK's example/aws/credential/plugincreds
// folder for a runnable example of this.
//
//   // Open plugin, and load it into the process.
//   p, err := plugin.Open("somefile.so")
//   if err != nil {
//   	return nil, err
//   }
//
//   // Create a new Credentials value which will source the provider's Retrieve
//   // and IsExpired functions from the plugin.
//   creds, err := plugincreds.NewCredentials(p)
//   if err != nil {
//   	return nil, err
//   }
//
//   // Example to configure a Session with the newly created credentials that
//   // will be sourced using the plugin's functionality.
//   sess := session.Must(session.NewSession(&aws.Config{
//   	Credentials:  creds,
//   }))
package plugincreds

import (
	"fmt"
	"plugin"

	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/credentials"
)

// ProviderSymbolName the symbol name the SDK will use to lookup the plugin
// provider value from.
const ProviderSymbolName = `GetAWSSDKCredentialProvider`

// ProviderName is the name this credentials provider will label any returned
// credentials Value with.
const ProviderName = `PluginCredentialsProvider`

const (
	// ErrCodeLookupSymbolError failed to lookup symbol
	ErrCodeLookupSymbolError = "LookupSymbolError"

	// ErrCodeInvalidSymbolError symbol invalid
	ErrCodeInvalidSymbolError = "InvalidSymbolError"

	// ErrCodePluginRetrieveNil Retrieve function was nil
	ErrCodePluginRetrieveNil = "PluginRetrieveNilError"

	// ErrCodePluginIsExpiredNil IsExpired Function was nil
	ErrCodePluginIsExpiredNil = "PluginIsExpiredNilError"

	// ErrCodePluginProviderRetrieve plugin provider's retrieve returned error
	ErrCodePluginProviderRetrieve = "PluginProviderRetrieveError"
)

// Provider is the credentials provider that will use the plugin provided
// Retrieve and IsExpired functions to retrieve credentials.
type Provider struct {
	RetrieveFn  func() (key, secret, token string, err error)
	IsExpiredFn func() bool
}

// NewCredentials returns a new Credentials loader using the plugin provider.
// If the symbol isn't found or is invalid in the plugin an error will be
// returned.
func NewCredentials(p *plugin.Plugin) (*credentials.Credentials, error) {
	retrieve, isExpired, err := GetPluginProviderFns(p)
	if err != nil {
		return nil, err
	}

	return credentials.NewCredentials(Provider{
		RetrieveFn:  retrieve,
		IsExpiredFn: isExpired,
	}), nil
}

// Retrieve will return the credentials Value if they were successfully retrieved
// from the underlying plugin provider. An error will be returned otherwise.
func (p Provider) Retrieve() (credentials.Value, error) {
	creds := credentials.Value{
		ProviderName: ProviderName,
	}

	k, s, t, err := p.RetrieveFn()
	if err != nil {
		return creds, awserr.New(ErrCodePluginProviderRetrieve,
			"failed to retrieve credentials with plugin provider", err)
	}

	creds.AccessKeyID = k
	creds.SecretAccessKey = s
	creds.SessionToken = t

	return creds, nil
}

// IsExpired will return the expired state of the underlying plugin provider.
func (p Provider) IsExpired() bool {
	return p.IsExpiredFn()
}

// GetPluginProviderFns returns the plugin's Retrieve and IsExpired functions
// returned by the plugin's credential provider getter.
//
// Uses ProviderSymbolName as the symbol name when lookup up the symbol. If you
// want to use a different symbol name, use GetPluginProviderFnsByName.
func GetPluginProviderFns(p *plugin.Plugin) (func() (key, secret, token string, err error), func() bool, error) {
	return GetPluginProviderFnsByName(p, ProviderSymbolName)
}

// GetPluginProviderFnsByName returns the plugin's Retrieve and IsExpired functions
// returned by the plugin's credential provider getter.
//
// Same as GetPluginProviderFns, but takes a custom symbolName to lookup with.
func GetPluginProviderFnsByName(p *plugin.Plugin, symbolName string) (func() (key, secret, token string, err error), func() bool, error) {
	sym, err := p.Lookup(symbolName)
	if err != nil {
		return nil, nil, awserr.New(ErrCodeLookupSymbolError,
			fmt.Sprintf("failed to lookup %s plugin provider symbol", symbolName), err)
	}

	fn, ok := sym.(func() (func() (key, secret, token string, err error), func() bool))
	if !ok {
		return nil, nil, awserr.New(ErrCodeInvalidSymbolError,
			fmt.Sprintf("symbol %T, does not match the 'func() (func() (key, secret, token string, err error), func() bool)'  type", sym), nil)
	}

	retrieveFn, isExpiredFn := fn()
	if retrieveFn == nil {
		return nil, nil, awserr.New(ErrCodePluginRetrieveNil,
			"the plugin provider retrieve function cannot be nil", nil)
	}
	if isExpiredFn == nil {
		return nil, nil, awserr.New(ErrCodePluginIsExpiredNil,
			"the plugin provider isExpired function cannot be nil", nil)
	}

	return retrieveFn, isExpiredFn, nil
}
