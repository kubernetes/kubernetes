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
	"context"
	"fmt"
	"net/http"
	"net/url"
	"time"

	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vapi/internal"
	"github.com/vmware/govmomi/vapi/rest"
)

// StorageBackings for Content Libraries
type StorageBackings struct {
	DatastoreID string `json:"datastore_id,omitempty"`
	Type        string `json:"type,omitempty"`
}

// Library  provides methods to create, read, update, delete, and enumerate libraries.
type Library struct {
	CreationTime     *time.Time        `json:"creation_time,omitempty"`
	Description      string            `json:"description,omitempty"`
	ID               string            `json:"id,omitempty"`
	LastModifiedTime *time.Time        `json:"last_modified_time,omitempty"`
	LastSyncTime     *time.Time        `json:"last_sync_time,omitempty"`
	Name             string            `json:"name,omitempty"`
	Storage          []StorageBackings `json:"storage_backings,omitempty"`
	Type             string            `json:"type,omitempty"`
	Version          string            `json:"version,omitempty"`
	Subscription     *Subscription     `json:"subscription_info,omitempty"`
	Publication      *Publication      `json:"publish_info,omitempty"`
}

// Subscription info
type Subscription struct {
	AuthenticationMethod string `json:"authentication_method"`
	AutomaticSyncEnabled *bool  `json:"automatic_sync_enabled,omitempty"`
	OnDemand             *bool  `json:"on_demand,omitempty"`
	Password             string `json:"password,omitempty"`
	SslThumbprint        string `json:"ssl_thumbprint,omitempty"`
	SubscriptionURL      string `json:"subscription_url,omitempty"`
	UserName             string `json:"user_name,omitempty"`
}

// Publication info
type Publication struct {
	AuthenticationMethod string `json:"authentication_method"`
	UserName             string `json:"user_name,omitempty"`
	Password             string `json:"password,omitempty"`
	CurrentPassword      string `json:"current_password,omitempty"`
	PersistJSON          *bool  `json:"persist_json_enabled,omitempty"`
	Published            *bool  `json:"published,omitempty"`
	PublishURL           string `json:"publish_url,omitempty"`
}

// SubscriberSummary as returned by ListSubscribers
type SubscriberSummary struct {
	LibraryID              string `json:"subscribed_library"`
	LibraryName            string `json:"subscribed_library_name"`
	SubscriptionID         string `json:"subscription"`
	LibraryVcenterHostname string `json:"subscribed_library_vcenter_hostname,omitempty"`
}

// Placement information used to place a virtual machine template
type Placement struct {
	ResourcePool string `json:"resource_pool,omitempty"`
	Host         string `json:"host,omitempty"`
	Folder       string `json:"folder,omitempty"`
	Cluster      string `json:"cluster,omitempty"`
	Network      string `json:"network,omitempty"`
}

// Vcenter contains information about the vCenter Server instance where a subscribed library associated with a subscription exists.
type Vcenter struct {
	Hostname   string `json:"hostname"`
	Port       int    `json:"https_port,omitempty"`
	ServerGUID string `json:"server_guid"`
}

// Subscriber contains the detailed info for a library subscriber.
type Subscriber struct {
	LibraryID       string     `json:"subscribed_library"`
	LibraryName     string     `json:"subscribed_library_name"`
	LibraryLocation string     `json:"subscribed_library_location"`
	Placement       *Placement `json:"subscribed_library_placement,omitempty"`
	Vcenter         *Vcenter   `json:"subscribed_library_vcenter,omitempty"`
}

// SubscriberLibrary is the specification for a subscribed library to be associated with a subscription.
type SubscriberLibrary struct {
	Target    string     `json:"target"`
	LibraryID string     `json:"subscribed_library,omitempty"`
	Location  string     `json:"location"`
	Vcenter   *Vcenter   `json:"vcenter,omitempty"`
	Placement *Placement `json:"placement,omitempty"`
}

// Patch merges updates from the given src.
func (l *Library) Patch(src *Library) {
	if src.Name != "" {
		l.Name = src.Name
	}
	if src.Description != "" {
		l.Description = src.Description
	}
	if src.Version != "" {
		l.Version = src.Version
	}
}

// Manager extends rest.Client, adding content library related methods.
type Manager struct {
	*rest.Client
}

// NewManager creates a new Manager instance with the given client.
func NewManager(client *rest.Client) *Manager {
	return &Manager{
		Client: client,
	}
}

// Find is the search criteria for finding libraries.
type Find struct {
	Name string `json:"name,omitempty"`
	Type string `json:"type,omitempty"`
}

// FindLibrary returns one or more libraries that match the provided search
// criteria.
//
// The provided name is case-insensitive.
//
// Either the name or type of library may be set to empty values in order
// to search for all libraries, all libraries with a specific name, regardless
// of type, or all libraries of a specified type.
func (c *Manager) FindLibrary(ctx context.Context, search Find) ([]string, error) {
	url := c.Resource(internal.LibraryPath).WithAction("find")
	spec := struct {
		Spec Find `json:"spec"`
	}{search}
	var res []string
	return res, c.Do(ctx, url.Request(http.MethodPost, spec), &res)
}

// CreateLibrary creates a new library with the given Type, Name,
// Description, and CategoryID.
func (c *Manager) CreateLibrary(ctx context.Context, library Library) (string, error) {
	spec := struct {
		Library Library `json:"create_spec"`
	}{library}
	path := internal.LocalLibraryPath
	if library.Type == "SUBSCRIBED" {
		path = internal.SubscribedLibraryPath
		sub := library.Subscription
		u, err := url.Parse(sub.SubscriptionURL)
		if err != nil {
			return "", err
		}
		if u.Scheme == "https" && sub.SslThumbprint == "" {
			thumbprint := c.Thumbprint(u.Host)
			if thumbprint == "" {
				t := c.DefaultTransport()
				if t.TLSClientConfig.InsecureSkipVerify {
					var info object.HostCertificateInfo
					_ = info.FromURL(u, t.TLSClientConfig)
					thumbprint = info.ThumbprintSHA1
				}
				sub.SslThumbprint = thumbprint
			}
		}
	}
	url := c.Resource(path)
	var res string
	return res, c.Do(ctx, url.Request(http.MethodPost, spec), &res)
}

// SyncLibrary syncs a subscribed library.
func (c *Manager) SyncLibrary(ctx context.Context, library *Library) error {
	path := internal.SubscribedLibraryPath
	url := c.Resource(path).WithID(library.ID).WithAction("sync")
	return c.Do(ctx, url.Request(http.MethodPost), nil)
}

// PublishLibrary publishes the library to specified subscriptions.
// If no subscriptions are specified, then publishes the library to all subscriptions.
func (c *Manager) PublishLibrary(ctx context.Context, library *Library, subscriptions []string) error {
	path := internal.LocalLibraryPath
	var spec internal.SubscriptionDestinationSpec
	for i := range subscriptions {
		spec.Subscriptions = append(spec.Subscriptions, internal.SubscriptionDestination{ID: subscriptions[i]})
	}
	url := c.Resource(path).WithID(library.ID).WithAction("publish")
	return c.Do(ctx, url.Request(http.MethodPost, spec), nil)
}

// UpdateLibrary can update one or both of the tag Description and Name fields.
func (c *Manager) UpdateLibrary(ctx context.Context, l *Library) error {
	spec := struct {
		Library `json:"update_spec"`
	}{
		Library{
			Name:        l.Name,
			Description: l.Description,
		},
	}
	url := c.Resource(internal.LibraryPath).WithID(l.ID)
	return c.Do(ctx, url.Request(http.MethodPatch, spec), nil)
}

// DeleteLibrary deletes an existing library.
func (c *Manager) DeleteLibrary(ctx context.Context, library *Library) error {
	path := internal.LocalLibraryPath
	if library.Type == "SUBSCRIBED" {
		path = internal.SubscribedLibraryPath
	}
	url := c.Resource(path).WithID(library.ID)
	return c.Do(ctx, url.Request(http.MethodDelete), nil)
}

// ListLibraries returns a list of all content library IDs in the system.
func (c *Manager) ListLibraries(ctx context.Context) ([]string, error) {
	url := c.Resource(internal.LibraryPath)
	var res []string
	return res, c.Do(ctx, url.Request(http.MethodGet), &res)
}

// GetLibraryByID returns information on a library for the given ID.
func (c *Manager) GetLibraryByID(ctx context.Context, id string) (*Library, error) {
	url := c.Resource(internal.LibraryPath).WithID(id)
	var res Library
	return &res, c.Do(ctx, url.Request(http.MethodGet), &res)
}

// GetLibraryByName returns information on a library for the given name.
func (c *Manager) GetLibraryByName(ctx context.Context, name string) (*Library, error) {
	// Lookup by name
	libraries, err := c.GetLibraries(ctx)
	if err != nil {
		return nil, err
	}
	for i := range libraries {
		if libraries[i].Name == name {
			return &libraries[i], nil
		}
	}
	return nil, fmt.Errorf("library name (%s) not found", name)
}

// GetLibraries returns a list of all content library details in the system.
func (c *Manager) GetLibraries(ctx context.Context) ([]Library, error) {
	ids, err := c.ListLibraries(ctx)
	if err != nil {
		return nil, fmt.Errorf("get libraries failed for: %s", err)
	}

	var libraries []Library
	for _, id := range ids {
		library, err := c.GetLibraryByID(ctx, id)
		if err != nil {
			return nil, fmt.Errorf("get library %s failed for %s", id, err)
		}

		libraries = append(libraries, *library)

	}
	return libraries, nil
}

// ListSubscribers lists the subscriptions of the published library.
func (c *Manager) ListSubscribers(ctx context.Context, library *Library) ([]SubscriberSummary, error) {
	url := c.Resource(internal.Subscriptions).WithParam("library", library.ID)
	var res []SubscriberSummary
	return res, c.Do(ctx, url.Request(http.MethodGet), &res)
}

// CreateSubscriber creates a subscription of the published library.
func (c *Manager) CreateSubscriber(ctx context.Context, library *Library, s SubscriberLibrary) (string, error) {
	var spec struct {
		Sub struct {
			SubscriberLibrary SubscriberLibrary `json:"subscribed_library"`
		} `json:"spec"`
	}
	spec.Sub.SubscriberLibrary = s
	url := c.Resource(internal.Subscriptions).WithID(library.ID)
	var res string
	return res, c.Do(ctx, url.Request(http.MethodPost, &spec), &res)
}

// GetSubscriber returns information about the specified subscriber of the published library.
func (c *Manager) GetSubscriber(ctx context.Context, library *Library, subscriber string) (*Subscriber, error) {
	id := internal.SubscriptionDestination{ID: subscriber}
	url := c.Resource(internal.Subscriptions).WithID(library.ID).WithAction("get")
	var res Subscriber
	return &res, c.Do(ctx, url.Request(http.MethodPost, &id), &res)
}

// DeleteSubscriber deletes the specified subscription of the published library.
// The subscribed library associated with the subscription will not be deleted.
func (c *Manager) DeleteSubscriber(ctx context.Context, library *Library, subscriber string) error {
	id := internal.SubscriptionDestination{ID: subscriber}
	url := c.Resource(internal.Subscriptions).WithID(library.ID).WithAction("delete")
	return c.Do(ctx, url.Request(http.MethodPost, &id), nil)
}
