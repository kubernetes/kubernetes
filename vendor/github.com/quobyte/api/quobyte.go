// Package quobyte represents a golang API for the Quobyte Storage System
package quobyte

import (
	"net/http"
)

// retry policy codes
const (
	RetryNever         string = "NEVER"
	RetryInteractive   string = "INTERACTIVE"
	RetryInfinitely    string = "INFINITELY"
	RetryOncePerTarget string = "ONCE_PER_TARGET"
)

type QuobyteClient struct {
	client         *http.Client
	url            string
	username       string
	password       string
	apiRetryPolicy string
}

func (client *QuobyteClient) SetAPIRetryPolicy(retry string) {
	client.apiRetryPolicy = retry
}

func (client *QuobyteClient) GetAPIRetryPolicy() string {
	return client.apiRetryPolicy
}

// NewQuobyteClient creates a new Quobyte API client
func NewQuobyteClient(url string, username string, password string) *QuobyteClient {
	return &QuobyteClient{
		client:         &http.Client{},
		url:            url,
		username:       username,
		password:       password,
		apiRetryPolicy: RetryInteractive,
	}
}

// CreateVolume creates a new Quobyte volume. Its root directory will be owned by given user and group
func (client QuobyteClient) CreateVolume(request *CreateVolumeRequest) (string, error) {
	var response volumeUUID
	if err := client.sendRequest("createVolume", request, &response); err != nil {
		return "", err
	}

	return response.VolumeUUID, nil
}

// ResolveVolumeNameToUUID resolves a volume name to a UUID
func (client *QuobyteClient) ResolveVolumeNameToUUID(volumeName, tenant string) (string, error) {
	request := &resolveVolumeNameRequest{
		VolumeName:   volumeName,
		TenantDomain: tenant,
	}
	var response volumeUUID
	if err := client.sendRequest("resolveVolumeName", request, &response); err != nil {
		return "", err
	}

	return response.VolumeUUID, nil
}

// DeleteVolume deletes a Quobyte volume
func (client *QuobyteClient) DeleteVolume(UUID string) error {
	return client.sendRequest(
		"deleteVolume",
		&volumeUUID{
			VolumeUUID: UUID,
		},
		nil)
}

// DeleteVolumeByName deletes a volume by a given name
func (client *QuobyteClient) DeleteVolumeByName(volumeName, tenant string) error {
	uuid, err := client.ResolveVolumeNameToUUID(volumeName, tenant)
	if err != nil {
		return err
	}

	return client.DeleteVolume(uuid)
}

// GetClientList returns a list of all active clients
func (client *QuobyteClient) GetClientList(tenant string) (GetClientListResponse, error) {
	request := &getClientListRequest{
		TenantDomain: tenant,
	}

	var response GetClientListResponse
	if err := client.sendRequest("getClientListRequest", request, &response); err != nil {
		return response, err
	}

	return response, nil
}

// SetVolumeQuota sets a Quota to the specified Volume
func (client *QuobyteClient) SetVolumeQuota(volumeUUID string, quotaSize uint64) error {
	request := &setQuotaRequest{
		Quotas: []*quota{
			&quota{
				Consumer: []*consumingEntity{
					&consumingEntity{
						Type:       "VOLUME",
						Identifier: volumeUUID,
					},
				},
				Limits: []*resource{
					&resource{
						Type:  "LOGICAL_DISK_SPACE",
						Value: quotaSize,
					},
				},
			},
		},
	}

	return client.sendRequest("setQuota", request, nil)
}

// GetTenant returns the Tenant configuration for all specified tenants
func (client *QuobyteClient) GetTenant(tenantIDs []string) (GetTenantResponse, error) {
	request := &getTenantRequest{TenantIDs: tenantIDs}

	var response GetTenantResponse
	err := client.sendRequest("getTenant", request, &response)
	if err != nil {
		return response, err
	}

	return response, nil
}

// GetTenantMap returns a map that contains all tenant names and there ID's
func (client *QuobyteClient) GetTenantMap() (map[string]string, error) {
	result := map[string]string{}
	response, err := client.GetTenant([]string{})

	if err != nil {
		return result, err
	}

	for _, tenant := range response.Tenants {
		result[tenant.Name] = tenant.TenantID
	}

	return result, nil
}

// SetTenant creates a Tenant with the specified name
func (client *QuobyteClient) SetTenant(tenantName string) (string, error) {
	request := &setTenantRequest{
		&TenantDomainConfiguration{
			Name: tenantName,
		},
		retryPolicy{client.GetAPIRetryPolicy()},
	}

	var response setTenantResponse
	err := client.sendRequest("setTenant", request, &response)
	if err != nil {
		return "", err
	}

	return response.TenantID, nil
}
