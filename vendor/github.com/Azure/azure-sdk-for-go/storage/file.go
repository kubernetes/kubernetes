package storage

import (
	"fmt"
	"net/http"
	"net/url"
)

// FileServiceClient contains operations for Microsoft Azure File Service.
type FileServiceClient struct {
	client Client
}

// pathForFileShare returns the URL path segment for a File Share resource
func pathForFileShare(name string) string {
	return fmt.Sprintf("/%s", name)
}

// CreateShare operation creates a new share under the specified account. If the
// share with the same name already exists, the operation fails.
//
// See https://msdn.microsoft.com/en-us/library/azure/dn167008.aspx
func (f FileServiceClient) CreateShare(name string) error {
	resp, err := f.createShare(name)
	if err != nil {
		return err
	}
	defer resp.body.Close()
	return checkRespCode(resp.statusCode, []int{http.StatusCreated})
}

// CreateShareIfNotExists creates a new share under the specified account if
// it does not exist. Returns true if container is newly created or false if
// container already exists.
//
// See https://msdn.microsoft.com/en-us/library/azure/dn167008.aspx
func (f FileServiceClient) CreateShareIfNotExists(name string) (bool, error) {
	resp, err := f.createShare(name)
	if resp != nil {
		defer resp.body.Close()
		if resp.statusCode == http.StatusCreated || resp.statusCode == http.StatusConflict {
			return resp.statusCode == http.StatusCreated, nil
		}
	}
	return false, err
}

// CreateShare creates a Azure File Share and returns its response
func (f FileServiceClient) createShare(name string) (*storageResponse, error) {
	if err := f.checkForStorageEmulator(); err != nil {
		return nil, err
	}
	uri := f.client.getEndpoint(fileServiceName, pathForFileShare(name), url.Values{"restype": {"share"}})
	headers := f.client.getStandardHeaders()
	return f.client.exec("PUT", uri, headers, nil)
}

// DeleteShare operation marks the specified share for deletion. The share
// and any files contained within it are later deleted during garbage
// collection.
//
// See https://msdn.microsoft.com/en-us/library/azure/dn689090.aspx
func (f FileServiceClient) DeleteShare(name string) error {
	resp, err := f.deleteShare(name)
	if err != nil {
		return err
	}
	defer resp.body.Close()
	return checkRespCode(resp.statusCode, []int{http.StatusAccepted})
}

// DeleteShareIfExists operation marks the specified share for deletion if it
// exists. The share and any files contained within it are later deleted during
// garbage collection. Returns true if share existed and deleted with this call,
// false otherwise.
//
// See https://msdn.microsoft.com/en-us/library/azure/dn689090.aspx
func (f FileServiceClient) DeleteShareIfExists(name string) (bool, error) {
	resp, err := f.deleteShare(name)
	if resp != nil {
		defer resp.body.Close()
		if resp.statusCode == http.StatusAccepted || resp.statusCode == http.StatusNotFound {
			return resp.statusCode == http.StatusAccepted, nil
		}
	}
	return false, err
}

// deleteShare makes the call to Delete Share operation endpoint and returns
// the response
func (f FileServiceClient) deleteShare(name string) (*storageResponse, error) {
	if err := f.checkForStorageEmulator(); err != nil {
		return nil, err
	}
	uri := f.client.getEndpoint(fileServiceName, pathForFileShare(name), url.Values{"restype": {"share"}})
	return f.client.exec("DELETE", uri, f.client.getStandardHeaders(), nil)
}

//checkForStorageEmulator determines if the client is setup for use with
//Azure Storage Emulator, and returns a relevant error
func (f FileServiceClient) checkForStorageEmulator() error {
	if f.client.accountName == StorageEmulatorAccountName {
		return fmt.Errorf("Error: File service is not currently supported by Azure Storage Emulator")
	}
	return nil
}
