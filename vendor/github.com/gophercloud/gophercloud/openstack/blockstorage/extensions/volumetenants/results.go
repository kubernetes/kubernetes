package volumetenants

// VolumeTenantExt is an extension to the base Volume object
type VolumeTenantExt struct {
	// TenantID is the id of the project that owns the volume.
	TenantID string `json:"os-vol-tenant-attr:tenant_id"`
}
