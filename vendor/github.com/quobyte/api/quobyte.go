// Package quobyte represents a golang API for the Quobyte Storage System
package quobyte

import "net/http"

type QuobyteClient struct {
	client   *http.Client
	url      string
	username string
	password string
}

// NewQuobyteClient creates a new Quobyte API client
func NewQuobyteClient(url string, username string, password string) *QuobyteClient {
	return &QuobyteClient{
		client:   &http.Client{},
		url:      url,
		username: username,
		password: password,
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
