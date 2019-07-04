package quobyte

type retryPolicy struct {
        RetryPolicy string `json:"retry,omitempty"`
}

// CreateVolumeRequest represents a CreateVolumeRequest
type CreateVolumeRequest struct {
        Name              string   `json:"name,omitempty"`
        RootUserID        string   `json:"root_user_id,omitempty"`
        RootGroupID       string   `json:"root_group_id,omitempty"`
        ReplicaDeviceIDS  []uint64 `json:"replica_device_ids,string,omitempty"`
        ConfigurationName string   `json:"configuration_name,omitempty"`
        AccessMode        uint32   `json:"access_mode,string,omitempty"`
        TenantID          string   `json:"tenant_id,omitempty"`
        retryPolicy
}

type resolveVolumeNameRequest struct {
        VolumeName   string `json:"volume_name,omitempty"`
        TenantDomain string `json:"tenant_domain,omitempty"`
        retryPolicy
}

type resolveTenantNameRequest struct {
	TenantName string `json:"tenant_name,omitempty"`
}

type resolveTenantNameResponse struct {
	TenantID string `json:"tenant_id,omitempty"`
}

type volumeUUID struct {
	VolumeUUID string `json:"volume_uuid,omitempty"`
}

type getClientListRequest struct {
        TenantDomain string `json:"tenant_domain,omitempty"`
        retryPolicy
}

type GetClientListResponse struct {
	Clients []Client `json:"client,omitempty"`
}

type Client struct {
	MountedUserName   string `json:"mount_user_name,omitempty"`
	MountedVolumeUUID string `json:"mounted_volume_uuid,omitempty"`
}

type consumingEntity struct {
	Type       string `json:"type,omitempty"`
	Identifier string `json:"identifier,omitempty"`
	TenantID   string `json:"tenant_id,omitempty"`
}

type resource struct {
	Type  string `json:"type,omitempty"`
	Value uint64 `json:"value,omitempty"`
}

type quota struct {
	ID           string             `json:"id,omitempty"`
	Consumer     []*consumingEntity `json:"consumer,omitempty"`
	Limits       []*resource        `json:"limits,omitempty"`
	Currentusage []*resource        `json:"current_usage,omitempty"`
}

type setQuotaRequest struct {
        Quotas []*quota `json:"quotas,omitempty"`
        retryPolicy
}

type getTenantRequest struct {
        TenantIDs []string `json:"tenant_id,omitempty"`
        retryPolicy
}

type GetTenantResponse struct {
	Tenants []*TenantDomainConfiguration `json:"tenant,omitempty"`
}

type TenantDomainConfiguration struct {
	TenantID          string                                   `json:"tenant_id,omitempty"`
	Name              string                                   `json:"name,omitempty"`
	RestrictToNetwork []string                                 `json:"restrict_to_network,omitempty"`
	VolumeAccess      []*TenantDomainConfigurationVolumeAccess `json:"volume_access,omitempty"`
}

type TenantDomainConfigurationVolumeAccess struct {
	VolumeUUID        string `json:"volume_uuid,omitempty"`
	RestrictToNetwork string `json:"restrict_to_network,omitempty"`
	ReadOnly          bool   `json:"read_only,omitempty"`
}

type setTenantRequest struct {
        Tenants *TenantDomainConfiguration `json:"tenant,omitempty"`
        retryPolicy
}

type setTenantResponse struct {
	TenantID string `json:"tenant_id,omitempty"`
}
