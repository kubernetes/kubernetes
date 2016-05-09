package servers

import (
	"crypto/rsa"
	"encoding/base64"
	"fmt"
	"net/url"
	"path"
	"reflect"

	"github.com/mitchellh/mapstructure"
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

type serverResult struct {
	gophercloud.Result
}

// Extract interprets any serverResult as a Server, if possible.
func (r serverResult) Extract() (*Server, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var response struct {
		Server Server `mapstructure:"server"`
	}

	config := &mapstructure.DecoderConfig{
		DecodeHook: toMapFromString,
		Result:     &response,
	}
	decoder, err := mapstructure.NewDecoder(config)
	if err != nil {
		return nil, err
	}

	err = decoder.Decode(r.Body)
	if err != nil {
		return nil, err
	}

	return &response.Server, nil
}

// CreateResult temporarily contains the response from a Create call.
type CreateResult struct {
	serverResult
}

// GetResult temporarily contains the response from a Get call.
type GetResult struct {
	serverResult
}

// UpdateResult temporarily contains the response from an Update call.
type UpdateResult struct {
	serverResult
}

// DeleteResult temporarily contains the response from a Delete call.
type DeleteResult struct {
	gophercloud.ErrResult
}

// RebuildResult temporarily contains the response from a Rebuild call.
type RebuildResult struct {
	serverResult
}

// ActionResult represents the result of server action operations, like reboot
type ActionResult struct {
	gophercloud.ErrResult
}

// RescueResult represents the result of a server rescue operation
type RescueResult struct {
	ActionResult
}

// CreateImageResult represents the result of an image creation operation
type CreateImageResult struct {
	gophercloud.Result
}

// GetPasswordResult represent the result of a get os-server-password operation.
type GetPasswordResult struct {
	gophercloud.Result
}

// ExtractPassword gets the encrypted password.
// If privateKey != nil the password is decrypted with the private key.
// If privateKey == nil the encrypted password is returned and can be decrypted with:
//   echo '<pwd>' | base64 -D | openssl rsautl -decrypt -inkey <private_key>
func (r GetPasswordResult) ExtractPassword(privateKey *rsa.PrivateKey) (string, error) {

	if r.Err != nil {
		return "", r.Err
	}

	var response struct {
		Password string `mapstructure:"password"`
	}

	err := mapstructure.Decode(r.Body, &response)
	if err == nil && privateKey != nil && response.Password != "" {
		return decryptPassword(response.Password, privateKey)
	}
	return response.Password, err
}

func decryptPassword(encryptedPassword string, privateKey *rsa.PrivateKey) (string, error) {
	b64EncryptedPassword := make([]byte, base64.StdEncoding.DecodedLen(len(encryptedPassword)))

	n, err := base64.StdEncoding.Decode(b64EncryptedPassword, []byte(encryptedPassword))
	if err != nil {
		return "", fmt.Errorf("Failed to base64 decode encrypted password: %s", err)
	}
	password, err := rsa.DecryptPKCS1v15(nil, privateKey, b64EncryptedPassword[0:n])
	if err != nil {
		return "", fmt.Errorf("Failed to decrypt password: %s", err)
	}

	return string(password), nil
}

// ExtractImageID gets the ID of the newly created server image from the header
func (res CreateImageResult) ExtractImageID() (string, error) {
	if res.Err != nil {
		return "", res.Err
	}
	// Get the image id from the header
	u, err := url.ParseRequestURI(res.Header.Get("Location"))
	if err != nil {
		return "", fmt.Errorf("Failed to parse the image id: %s", err.Error())
	}
	imageId := path.Base(u.Path)
	if imageId == "." || imageId == "/" {
		return "", fmt.Errorf("Failed to parse the ID of newly created image: %s", u)
	}
	return imageId, nil
}

// Extract interprets any RescueResult as an AdminPass, if possible.
func (r RescueResult) Extract() (string, error) {
	if r.Err != nil {
		return "", r.Err
	}

	var response struct {
		AdminPass string `mapstructure:"adminPass"`
	}

	err := mapstructure.Decode(r.Body, &response)
	return response.AdminPass, err
}

// Server exposes only the standard OpenStack fields corresponding to a given server on the user's account.
type Server struct {
	// ID uniquely identifies this server amongst all other servers, including those not accessible to the current tenant.
	ID string

	// TenantID identifies the tenant owning this server resource.
	TenantID string `mapstructure:"tenant_id"`

	// UserID uniquely identifies the user account owning the tenant.
	UserID string `mapstructure:"user_id"`

	// Name contains the human-readable name for the server.
	Name string

	// Updated and Created contain ISO-8601 timestamps of when the state of the server last changed, and when it was created.
	Updated string
	Created string

	HostID string

	// Status contains the current operational status of the server, such as IN_PROGRESS or ACTIVE.
	Status string

	// Progress ranges from 0..100.
	// A request made against the server completes only once Progress reaches 100.
	Progress int

	// AccessIPv4 and AccessIPv6 contain the IP addresses of the server, suitable for remote access for administration.
	AccessIPv4, AccessIPv6 string

	// Image refers to a JSON object, which itself indicates the OS image used to deploy the server.
	Image map[string]interface{}

	// Flavor refers to a JSON object, which itself indicates the hardware configuration of the deployed server.
	Flavor map[string]interface{}

	// Addresses includes a list of all IP addresses assigned to the server, keyed by pool.
	Addresses map[string]interface{}

	// Metadata includes a list of all user-specified key-value pairs attached to the server.
	Metadata map[string]interface{}

	// Links includes HTTP references to the itself, useful for passing along to other APIs that might want a server reference.
	Links []interface{}

	// KeyName indicates which public key was injected into the server on launch.
	KeyName string `json:"key_name" mapstructure:"key_name"`

	// AdminPass will generally be empty ("").  However, it will contain the administrative password chosen when provisioning a new server without a set AdminPass setting in the first place.
	// Note that this is the ONLY time this field will be valid.
	AdminPass string `json:"adminPass" mapstructure:"adminPass"`

	// SecurityGroups includes the security groups that this instance has applied to it
	SecurityGroups []map[string]interface{} `json:"security_groups" mapstructure:"security_groups"`
}

// ServerPage abstracts the raw results of making a List() request against the API.
// As OpenStack extensions may freely alter the response bodies of structures returned to the client, you may only safely access the
// data provided through the ExtractServers call.
type ServerPage struct {
	pagination.LinkedPageBase
}

// IsEmpty returns true if a page contains no Server results.
func (page ServerPage) IsEmpty() (bool, error) {
	servers, err := ExtractServers(page)
	if err != nil {
		return true, err
	}
	return len(servers) == 0, nil
}

// NextPageURL uses the response's embedded link reference to navigate to the next page of results.
func (page ServerPage) NextPageURL() (string, error) {
	type resp struct {
		Links []gophercloud.Link `mapstructure:"servers_links"`
	}

	var r resp
	err := mapstructure.Decode(page.Body, &r)
	if err != nil {
		return "", err
	}

	return gophercloud.ExtractNextURL(r.Links)
}

// ExtractServers interprets the results of a single page from a List() call, producing a slice of Server entities.
func ExtractServers(page pagination.Page) ([]Server, error) {
	casted := page.(ServerPage).Body

	var response struct {
		Servers []Server `mapstructure:"servers"`
	}

	config := &mapstructure.DecoderConfig{
		DecodeHook: toMapFromString,
		Result:     &response,
	}
	decoder, err := mapstructure.NewDecoder(config)
	if err != nil {
		return nil, err
	}

	err = decoder.Decode(casted)

	return response.Servers, err
}

// MetadataResult contains the result of a call for (potentially) multiple key-value pairs.
type MetadataResult struct {
	gophercloud.Result
}

// GetMetadataResult temporarily contains the response from a metadata Get call.
type GetMetadataResult struct {
	MetadataResult
}

// ResetMetadataResult temporarily contains the response from a metadata Reset call.
type ResetMetadataResult struct {
	MetadataResult
}

// UpdateMetadataResult temporarily contains the response from a metadata Update call.
type UpdateMetadataResult struct {
	MetadataResult
}

// MetadatumResult contains the result of a call for individual a single key-value pair.
type MetadatumResult struct {
	gophercloud.Result
}

// GetMetadatumResult temporarily contains the response from a metadatum Get call.
type GetMetadatumResult struct {
	MetadatumResult
}

// CreateMetadatumResult temporarily contains the response from a metadatum Create call.
type CreateMetadatumResult struct {
	MetadatumResult
}

// DeleteMetadatumResult temporarily contains the response from a metadatum Delete call.
type DeleteMetadatumResult struct {
	gophercloud.ErrResult
}

// Extract interprets any MetadataResult as a Metadata, if possible.
func (r MetadataResult) Extract() (map[string]string, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var response struct {
		Metadata map[string]string `mapstructure:"metadata"`
	}

	err := mapstructure.Decode(r.Body, &response)
	return response.Metadata, err
}

// Extract interprets any MetadatumResult as a Metadatum, if possible.
func (r MetadatumResult) Extract() (map[string]string, error) {
	if r.Err != nil {
		return nil, r.Err
	}

	var response struct {
		Metadatum map[string]string `mapstructure:"meta"`
	}

	err := mapstructure.Decode(r.Body, &response)
	return response.Metadatum, err
}

func toMapFromString(from reflect.Kind, to reflect.Kind, data interface{}) (interface{}, error) {
	if (from == reflect.String) && (to == reflect.Map) {
		return map[string]interface{}{}, nil
	}
	return data, nil
}

// Address represents an IP address.
type Address struct {
	Version int    `mapstructure:"version"`
	Address string `mapstructure:"addr"`
}

// AddressPage abstracts the raw results of making a ListAddresses() request against the API.
// As OpenStack extensions may freely alter the response bodies of structures returned
// to the client, you may only safely access the data provided through the ExtractAddresses call.
type AddressPage struct {
	pagination.SinglePageBase
}

// IsEmpty returns true if an AddressPage contains no networks.
func (r AddressPage) IsEmpty() (bool, error) {
	addresses, err := ExtractAddresses(r)
	if err != nil {
		return true, err
	}
	return len(addresses) == 0, nil
}

// ExtractAddresses interprets the results of a single page from a ListAddresses() call,
// producing a map of addresses.
func ExtractAddresses(page pagination.Page) (map[string][]Address, error) {
	casted := page.(AddressPage).Body

	var response struct {
		Addresses map[string][]Address `mapstructure:"addresses"`
	}

	err := mapstructure.Decode(casted, &response)
	if err != nil {
		return nil, err
	}

	return response.Addresses, err
}

// NetworkAddressPage abstracts the raw results of making a ListAddressesByNetwork() request against the API.
// As OpenStack extensions may freely alter the response bodies of structures returned
// to the client, you may only safely access the data provided through the ExtractAddresses call.
type NetworkAddressPage struct {
	pagination.SinglePageBase
}

// IsEmpty returns true if a NetworkAddressPage contains no addresses.
func (r NetworkAddressPage) IsEmpty() (bool, error) {
	addresses, err := ExtractNetworkAddresses(r)
	if err != nil {
		return true, err
	}
	return len(addresses) == 0, nil
}

// ExtractNetworkAddresses interprets the results of a single page from a ListAddressesByNetwork() call,
// producing a slice of addresses.
func ExtractNetworkAddresses(page pagination.Page) ([]Address, error) {
	casted := page.(NetworkAddressPage).Body

	var response map[string][]Address
	err := mapstructure.Decode(casted, &response)
	if err != nil {
		return nil, err
	}

	var key string
	for k := range response {
		key = k
	}

	return response[key], err
}
