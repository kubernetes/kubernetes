/*
Copyright (c) 2018 VMware, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package library

import (
	"bufio"
	"context"
	"io"
	"net/http"
	"strings"

	"github.com/vmware/govmomi/vapi/internal"
	"github.com/vmware/govmomi/vapi/rest"
	"github.com/vmware/govmomi/vim25/soap"
)

// TransferEndpoint provides information on the source of a library item file.
type TransferEndpoint struct {
	URI                      string `json:"uri,omitempty"`
	SSLCertificateThumbprint string `json:"ssl_certificate_thumbprint,omitempty"`
}

// UpdateFile is the specification for the updatesession
// operations file:add, file:get, and file:list.
type UpdateFile struct {
	BytesTransferred int64                    `json:"bytes_transferred,omitempty"`
	Checksum         *Checksum                `json:"checksum_info,omitempty"`
	ErrorMessage     *rest.LocalizableMessage `json:"error_message,omitempty"`
	Name             string                   `json:"name"`
	Size             int64                    `json:"size,omitempty"`
	SourceEndpoint   *TransferEndpoint        `json:"source_endpoint,omitempty"`
	SourceType       string                   `json:"source_type"`
	Status           string                   `json:"status,omitempty"`
	UploadEndpoint   *TransferEndpoint        `json:"upload_endpoint,omitempty"`
}

// AddLibraryItemFile adds a file
func (c *Manager) AddLibraryItemFile(ctx context.Context, sessionID string, updateFile UpdateFile) (*UpdateFile, error) {
	url := c.Resource(internal.LibraryItemUpdateSessionFile).WithID(sessionID).WithAction("add")
	spec := struct {
		FileSpec UpdateFile `json:"file_spec"`
	}{updateFile}
	var res UpdateFile
	err := c.Do(ctx, url.Request(http.MethodPost, spec), &res)
	if err != nil {
		return nil, err
	}
	if res.Status == "ERROR" {
		return nil, res.ErrorMessage
	}
	return &res, nil
}

// AddLibraryItemFileFromURI adds a file from a remote URI.
func (c *Manager) AddLibraryItemFileFromURI(
	ctx context.Context,
	sessionID, fileName, uri string) (*UpdateFile, error) {

	n, fingerprint, err := c.getContentLengthAndFingerprint(ctx, uri)
	if err != nil {
		return nil, err
	}

	info, err := c.AddLibraryItemFile(ctx, sessionID, UpdateFile{
		Name:       fileName,
		SourceType: "PULL",
		Size:       n,
		SourceEndpoint: &TransferEndpoint{
			URI:                      uri,
			SSLCertificateThumbprint: fingerprint,
		},
	})
	if err != nil {
		return nil, err
	}

	return info, c.CompleteLibraryItemUpdateSession(ctx, sessionID)
}

// GetLibraryItemUpdateSessionFile retrieves information about a specific file
// that is a part of an update session.
func (c *Manager) GetLibraryItemUpdateSessionFile(ctx context.Context, sessionID string, fileName string) (*UpdateFile, error) {
	url := c.Resource(internal.LibraryItemUpdateSessionFile).WithID(sessionID).WithAction("get")
	spec := struct {
		Name string `json:"file_name"`
	}{fileName}
	var res UpdateFile
	return &res, c.Do(ctx, url.Request(http.MethodPost, spec), &res)
}

// getContentLengthAndFingerprint gets the number of bytes returned
// by the URI as well as the SHA1 fingerprint of the peer certificate
// if the URI's scheme is https.
func (c *Manager) getContentLengthAndFingerprint(
	ctx context.Context, uri string) (int64, string, error) {
	resp, err := c.Head(uri)
	if err != nil {
		return 0, "", err
	}
	if resp.TLS == nil || len(resp.TLS.PeerCertificates) == 0 {
		return resp.ContentLength, "", nil
	}
	fingerprint := c.Thumbprint(resp.Request.URL.Host)
	if fingerprint == "" {
		if c.DefaultTransport().TLSClientConfig.InsecureSkipVerify {
			fingerprint = soap.ThumbprintSHA1(resp.TLS.PeerCertificates[0])
		}
	}
	return resp.ContentLength, fingerprint, nil
}

// ReadManifest converts an ovf manifest to a map of file name -> Checksum.
func ReadManifest(m io.Reader) (map[string]*Checksum, error) {
	// expected format: openssl sha1 *.{ovf,vmdk}
	c := make(map[string]*Checksum)

	scanner := bufio.NewScanner(m)
	for scanner.Scan() {
		line := strings.SplitN(scanner.Text(), ")=", 2)
		if len(line) != 2 {
			continue
		}
		name := strings.SplitN(line[0], "(", 2)
		if len(name) != 2 {
			continue
		}
		sum := &Checksum{
			Algorithm: strings.TrimSpace(name[0]),
			Checksum:  strings.TrimSpace(line[1]),
		}
		c[name[1]] = sum
	}

	return c, scanner.Err()
}
