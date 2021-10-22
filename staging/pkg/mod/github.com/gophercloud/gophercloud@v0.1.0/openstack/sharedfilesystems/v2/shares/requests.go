package shares

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// CreateOptsBuilder allows extensions to add additional parameters to the
// Create request.
type CreateOptsBuilder interface {
	ToShareCreateMap() (map[string]interface{}, error)
}

// CreateOpts contains the options for create a Share. This object is
// passed to shares.Create(). For more information about these parameters,
// please refer to the Share object, or the shared file systems API v2
// documentation
type CreateOpts struct {
	// Defines the share protocol to use
	ShareProto string `json:"share_proto" required:"true"`
	// Size in GB
	Size int `json:"size" required:"true"`
	// Defines the share name
	Name string `json:"name,omitempty"`
	// Share description
	Description string `json:"description,omitempty"`
	// DisplayName is equivalent to Name. The API supports using both
	// This is an inherited attribute from the block storage API
	DisplayName string `json:"display_name,omitempty"`
	// DisplayDescription is equivalent to Description. The API supports using both
	// This is an inherited attribute from the block storage API
	DisplayDescription string `json:"display_description,omitempty"`
	// ShareType defines the sharetype. If omitted, a default share type is used
	ShareType string `json:"share_type,omitempty"`
	// VolumeType is deprecated but supported. Either ShareType or VolumeType can be used
	VolumeType string `json:"volume_type,omitempty"`
	// The UUID from which to create a share
	SnapshotID string `json:"snapshot_id,omitempty"`
	// Determines whether or not the share is public
	IsPublic *bool `json:"is_public,omitempty"`
	// Key value pairs of user defined metadata
	Metadata map[string]string `json:"metadata,omitempty"`
	// The UUID of the share network to which the share belongs to
	ShareNetworkID string `json:"share_network_id,omitempty"`
	// The UUID of the consistency group to which the share belongs to
	ConsistencyGroupID string `json:"consistency_group_id,omitempty"`
	// The availability zone of the share
	AvailabilityZone string `json:"availability_zone,omitempty"`
}

// ToShareCreateMap assembles a request body based on the contents of a
// CreateOpts.
func (opts CreateOpts) ToShareCreateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "share")
}

// Create will create a new Share based on the values in CreateOpts. To extract
// the Share object from the response, call the Extract method on the
// CreateResult.
func Create(client *gophercloud.ServiceClient, opts CreateOptsBuilder) (r CreateResult) {
	b, err := opts.ToShareCreateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Post(createURL(client), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200, 201},
	})
	return
}

// ListOpts holds options for listing Shares. It is passed to the
// shares.List function.
type ListOpts struct {
	// (Admin only). Defines whether to list the requested resources for all projects.
	AllTenants bool `q:"all_tenants"`
	// The share name.
	Name string `q:"name"`
	// Filters by a share status.
	Status string `q:"status"`
	// The UUID of the share server.
	ShareServerID string `q:"share_server_id"`
	// One or more metadata key and value pairs as a dictionary of strings.
	Metadata map[string]string `q:"metadata"`
	// The extra specifications for the share type.
	ExtraSpecs map[string]string `q:"extra_specs"`
	// The UUID of the share type.
	ShareTypeID string `q:"share_type_id"`
	// The maximum number of shares to return.
	Limit int `q:"limit"`
	// The offset to define start point of share or share group listing.
	Offset int `q:"offset"`
	// The key to sort a list of shares.
	SortKey string `q:"sort_key"`
	// The direction to sort a list of shares.
	SortDir string `q:"sort_dir"`
	// The UUID of the share’s base snapshot to filter the request based on.
	SnapshotID string `q:"snapshot_id"`
	// The share host name.
	Host string `q:"host"`
	// The share network ID.
	ShareNetworkID string `q:"share_network_id"`
	// The UUID of the project in which the share was created. Useful with all_tenants parameter.
	ProjectID string `q:"project_id"`
	// The level of visibility for the share.
	IsPublic *bool `q:"is_public"`
	// The UUID of a share group to filter resource.
	ShareGroupID string `q:"share_group_id"`
	// The export location UUID that can be used to filter shares or share instances.
	ExportLocationID string `q:"export_location_id"`
	// The export location path that can be used to filter shares or share instances.
	ExportLocationPath string `q:"export_location_path"`
	// The name pattern that can be used to filter shares, share snapshots, share networks or share groups.
	NamePattern string `q:"name~"`
	// The description pattern that can be used to filter shares, share snapshots, share networks or share groups.
	DescriptionPattern string `q:"description~"`
	// Whether to show count in API response or not, default is False.
	WithCount bool `q:"with_count"`
	// DisplayName is equivalent to Name. The API supports using both
	// This is an inherited attribute from the block storage API
	DisplayName string `q:"display_name"`
	// Equivalent to NamePattern.
	DisplayNamePattern string `q:"display_name~"`
	// VolumeTypeID is deprecated but supported. Either ShareTypeID or VolumeTypeID can be used
	VolumeTypeID string `q:"volume_type_id"`
	// The UUID of the share group snapshot.
	ShareGroupSnapshotID string `q:"share_group_snapshot_id"`
	// DisplayDescription is equivalent to Description. The API supports using both
	// This is an inherited attribute from the block storage API
	DisplayDescription string `q:"display_description"`
	// Equivalent to DescriptionPattern
	DisplayDescriptionPattern string `q:"display_description~"`
}

// ListOptsBuilder allows extensions to add additional parameters to the List
// request.
type ListOptsBuilder interface {
	ToShareListQuery() (string, error)
}

// ToShareListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToShareListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
}

// ListDetail returns []Share optionally limited by the conditions provided in ListOpts.
func ListDetail(client *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	url := listDetailURL(client)
	if opts != nil {
		query, err := opts.ToShareListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}

	return pagination.NewPager(client, url, func(r pagination.PageResult) pagination.Page {
		p := SharePage{pagination.MarkerPageBase{PageResult: r}}
		p.MarkerPageBase.Owner = p
		return p
	})
}

// Delete will delete an existing Share with the given UUID.
func Delete(client *gophercloud.ServiceClient, id string) (r DeleteResult) {
	_, r.Err = client.Delete(deleteURL(client, id), nil)
	return
}

// Get will get a single share with given UUID
func Get(client *gophercloud.ServiceClient, id string) (r GetResult) {
	_, r.Err = client.Get(getURL(client, id), &r.Body, nil)
	return
}

// GetExportLocations will get shareID's export locations.
// Client must have Microversion set; minimum supported microversion for GetExportLocations is 2.14.
func GetExportLocations(client *gophercloud.ServiceClient, id string) (r GetExportLocationsResult) {
	_, r.Err = client.Get(getExportLocationsURL(client, id), &r.Body, nil)
	return
}

// GrantAccessOptsBuilder allows extensions to add additional parameters to the
// GrantAccess request.
type GrantAccessOptsBuilder interface {
	ToGrantAccessMap() (map[string]interface{}, error)
}

// GrantAccessOpts contains the options for creation of an GrantAccess request.
// For more information about these parameters, please, refer to the shared file systems API v2,
// Share Actions, Grant Access documentation
type GrantAccessOpts struct {
	// The access rule type that can be "ip", "cert" or "user".
	AccessType string `json:"access_type"`
	// The value that defines the access that can be a valid format of IP, cert or user.
	AccessTo string `json:"access_to"`
	// The access level to the share is either "rw" or "ro".
	AccessLevel string `json:"access_level"`
}

// ToGrantAccessMap assembles a request body based on the contents of a
// GrantAccessOpts.
func (opts GrantAccessOpts) ToGrantAccessMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "allow_access")
}

// GrantAccess will grant access to a Share based on the values in GrantAccessOpts. To extract
// the GrantAccess object from the response, call the Extract method on the GrantAccessResult.
// Client must have Microversion set; minimum supported microversion for GrantAccess is 2.7.
func GrantAccess(client *gophercloud.ServiceClient, id string, opts GrantAccessOptsBuilder) (r GrantAccessResult) {
	b, err := opts.ToGrantAccessMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Post(grantAccessURL(client, id), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}

// RevokeAccessOptsBuilder allows extensions to add additional parameters to the
// RevokeAccess request.
type RevokeAccessOptsBuilder interface {
	ToRevokeAccessMap() (map[string]interface{}, error)
}

// RevokeAccessOpts contains the options for creation of a RevokeAccess request.
// For more information about these parameters, please, refer to the shared file systems API v2,
// Share Actions, Revoke Access documentation
type RevokeAccessOpts struct {
	AccessID string `json:"access_id"`
}

// ToRevokeAccessMap assembles a request body based on the contents of a
// RevokeAccessOpts.
func (opts RevokeAccessOpts) ToRevokeAccessMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "deny_access")
}

// RevokeAccess will revoke an existing access to a Share based on the values in RevokeAccessOpts.
// RevokeAccessResult contains only the error. To extract it, call the ExtractErr method on
// the RevokeAccessResult. Client must have Microversion set; minimum supported microversion
// for RevokeAccess is 2.7.
func RevokeAccess(client *gophercloud.ServiceClient, id string, opts RevokeAccessOptsBuilder) (r RevokeAccessResult) {
	b, err := opts.ToRevokeAccessMap()
	if err != nil {
		r.Err = err
		return
	}

	_, r.Err = client.Post(revokeAccessURL(client, id), b, nil, &gophercloud.RequestOpts{
		OkCodes: []int{200, 202},
	})

	return
}

// ListAccessRights lists all access rules assigned to a Share based on its id. To extract
// the AccessRight slice from the response, call the Extract method on the ListAccessRightsResult.
// Client must have Microversion set; minimum supported microversion for ListAccessRights is 2.7.
func ListAccessRights(client *gophercloud.ServiceClient, id string) (r ListAccessRightsResult) {
	requestBody := map[string]interface{}{"access_list": nil}
	_, r.Err = client.Post(listAccessRightsURL(client, id), requestBody, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}

// ExtendOptsBuilder allows extensions to add additional parameters to the
// Extend request.
type ExtendOptsBuilder interface {
	ToShareExtendMap() (map[string]interface{}, error)
}

// ExtendOpts contains options for extending a Share.
// For more information about these parameters, please, refer to the shared file systems API v2,
// Share Actions, Extend share documentation
type ExtendOpts struct {
	// New size in GBs.
	NewSize int `json:"new_size"`
}

// ToShareExtendMap assembles a request body based on the contents of a
// ExtendOpts.
func (opts ExtendOpts) ToShareExtendMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "extend")
}

// Extend will extend the capacity of an existing share. ExtendResult contains only the error.
// To extract it, call the ExtractErr method on the ExtendResult.
// Client must have Microversion set; minimum supported microversion for Extend is 2.7.
func Extend(client *gophercloud.ServiceClient, id string, opts ExtendOptsBuilder) (r ExtendResult) {
	b, err := opts.ToShareExtendMap()
	if err != nil {
		r.Err = err
		return
	}

	_, r.Err = client.Post(extendURL(client, id), b, nil, &gophercloud.RequestOpts{
		OkCodes: []int{202},
	})

	return
}

// ShrinkOptsBuilder allows extensions to add additional parameters to the
// Shrink request.
type ShrinkOptsBuilder interface {
	ToShareShrinkMap() (map[string]interface{}, error)
}

// ShrinkOpts contains options for shrinking a Share.
// For more information about these parameters, please, refer to the shared file systems API v2,
// Share Actions, Shrink share documentation
type ShrinkOpts struct {
	// New size in GBs.
	NewSize int `json:"new_size"`
}

// ToShareShrinkMap assembles a request body based on the contents of a
// ShrinkOpts.
func (opts ShrinkOpts) ToShareShrinkMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "shrink")
}

// Shrink will shrink the capacity of an existing share. ShrinkResult contains only the error.
// To extract it, call the ExtractErr method on the ShrinkResult.
// Client must have Microversion set; minimum supported microversion for Shrink is 2.7.
func Shrink(client *gophercloud.ServiceClient, id string, opts ShrinkOptsBuilder) (r ShrinkResult) {
	b, err := opts.ToShareShrinkMap()
	if err != nil {
		r.Err = err
		return
	}

	_, r.Err = client.Post(shrinkURL(client, id), b, nil, &gophercloud.RequestOpts{
		OkCodes: []int{202},
	})

	return
}

// UpdateOptsBuilder allows extensions to add additional parameters to the
// Update request.
type UpdateOptsBuilder interface {
	ToShareUpdateMap() (map[string]interface{}, error)
}

// UpdateOpts contain options for updating an existing Share. This object is passed
// to the share.Update function. For more information about the parameters, see
// the Share object.
type UpdateOpts struct {
	// Share name. Manila share update logic doesn't have a "name" alias.
	DisplayName *string `json:"display_name,omitempty"`
	// Share description. Manila share update logic doesn't have a "description" alias.
	DisplayDescription *string `json:"display_description,omitempty"`
	// Determines whether or not the share is public
	IsPublic *bool `json:"is_public,omitempty"`
}

// ToShareUpdateMap assembles a request body based on the contents of an
// UpdateOpts.
func (opts UpdateOpts) ToShareUpdateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "share")
}

// Update will update the Share with provided information. To extract the updated
// Share from the response, call the Extract method on the UpdateResult.
func Update(client *gophercloud.ServiceClient, id string, opts UpdateOptsBuilder) (r UpdateResult) {
	b, err := opts.ToShareUpdateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Put(updateURL(client, id), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}
