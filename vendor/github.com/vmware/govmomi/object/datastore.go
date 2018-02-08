/*
Copyright (c) 2015-2016 VMware, Inc. All Rights Reserved.

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

package object

import (
	"fmt"
	"io"
	"math/rand"
	"os"
	"path"
	"strings"

	"context"
	"net/http"
	"net/url"

	"github.com/vmware/govmomi/property"
	"github.com/vmware/govmomi/session"
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

// DatastoreNoSuchDirectoryError is returned when a directory could not be found.
type DatastoreNoSuchDirectoryError struct {
	verb    string
	subject string
}

func (e DatastoreNoSuchDirectoryError) Error() string {
	return fmt.Sprintf("cannot %s '%s': No such directory", e.verb, e.subject)
}

// DatastoreNoSuchFileError is returned when a file could not be found.
type DatastoreNoSuchFileError struct {
	verb    string
	subject string
}

func (e DatastoreNoSuchFileError) Error() string {
	return fmt.Sprintf("cannot %s '%s': No such file", e.verb, e.subject)
}

type Datastore struct {
	Common

	DatacenterPath string
}

func NewDatastore(c *vim25.Client, ref types.ManagedObjectReference) *Datastore {
	return &Datastore{
		Common: NewCommon(c, ref),
	}
}

func (d Datastore) Path(path string) string {
	return (&DatastorePath{
		Datastore: d.Name(),
		Path:      path,
	}).String()
}

// NewURL constructs a url.URL with the given file path for datastore access over HTTP.
func (d Datastore) NewURL(path string) *url.URL {
	u := d.c.URL()

	return &url.URL{
		Scheme: u.Scheme,
		Host:   u.Host,
		Path:   fmt.Sprintf("/folder/%s", path),
		RawQuery: url.Values{
			"dcPath": []string{d.DatacenterPath},
			"dsName": []string{d.Name()},
		}.Encode(),
	}
}

// URL is deprecated, use NewURL instead.
func (d Datastore) URL(ctx context.Context, dc *Datacenter, path string) (*url.URL, error) {
	return d.NewURL(path), nil
}

func (d Datastore) Browser(ctx context.Context) (*HostDatastoreBrowser, error) {
	var do mo.Datastore

	err := d.Properties(ctx, d.Reference(), []string{"browser"}, &do)
	if err != nil {
		return nil, err
	}

	return NewHostDatastoreBrowser(d.c, do.Browser), nil
}

func (d Datastore) useServiceTicket() bool {
	// If connected to workstation, service ticketing not supported
	// If connected to ESX, service ticketing not needed
	if !d.c.IsVC() {
		return false
	}

	key := "GOVMOMI_USE_SERVICE_TICKET"

	val := d.c.URL().Query().Get(key)
	if val == "" {
		val = os.Getenv(key)
	}

	if val == "1" || val == "true" {
		return true
	}

	return false
}

func (d Datastore) useServiceTicketHostName(name string) bool {
	// No need if talking directly to ESX.
	if !d.c.IsVC() {
		return false
	}

	// If version happens to be < 5.1
	if name == "" {
		return false
	}

	// If the HostSystem is using DHCP on a network without dynamic DNS,
	// HostSystem.Config.Network.DnsConfig.HostName is set to "localhost" by default.
	// This resolves to "localhost.localdomain" by default via /etc/hosts on ESX.
	// In that case, we will stick with the HostSystem.Name which is the IP address that
	// was used to connect the host to VC.
	if name == "localhost.localdomain" {
		return false
	}

	// Still possible to have HostName that don't resolve via DNS,
	// so we default to false.
	key := "GOVMOMI_USE_SERVICE_TICKET_HOSTNAME"

	val := d.c.URL().Query().Get(key)
	if val == "" {
		val = os.Getenv(key)
	}

	if val == "1" || val == "true" {
		return true
	}

	return false
}

type datastoreServiceTicketHostKey struct{}

// HostContext returns a Context where the given host will be used for datastore HTTP access
// via the ServiceTicket method.
func (d Datastore) HostContext(ctx context.Context, host *HostSystem) context.Context {
	return context.WithValue(ctx, datastoreServiceTicketHostKey{}, host)
}

// ServiceTicket obtains a ticket via AcquireGenericServiceTicket and returns it an http.Cookie with the url.URL
// that can be used along with the ticket cookie to access the given path.  An host is chosen at random unless the
// the given Context was created with a specific host via the HostContext method.
func (d Datastore) ServiceTicket(ctx context.Context, path string, method string) (*url.URL, *http.Cookie, error) {
	u := d.NewURL(path)

	host, ok := ctx.Value(datastoreServiceTicketHostKey{}).(*HostSystem)

	if !ok {
		if !d.useServiceTicket() {
			return u, nil, nil
		}

		hosts, err := d.AttachedHosts(ctx)
		if err != nil {
			return nil, nil, err
		}

		if len(hosts) == 0 {
			// Fallback to letting vCenter choose a host
			return u, nil, nil
		}

		// Pick a random attached host
		host = hosts[rand.Intn(len(hosts))]
	}

	ips, err := host.ManagementIPs(ctx)
	if err != nil {
		return nil, nil, err
	}

	if len(ips) > 0 {
		// prefer a ManagementIP
		u.Host = ips[0].String()
	} else {
		// fallback to inventory name
		u.Host, err = host.ObjectName(ctx)
		if err != nil {
			return nil, nil, err
		}
	}

	// VC datacenter path will not be valid against ESX
	q := u.Query()
	delete(q, "dcPath")
	u.RawQuery = q.Encode()

	spec := types.SessionManagerHttpServiceRequestSpec{
		Url: u.String(),
		// See SessionManagerHttpServiceRequestSpecMethod enum
		Method: fmt.Sprintf("http%s%s", method[0:1], strings.ToLower(method[1:])),
	}

	sm := session.NewManager(d.Client())

	ticket, err := sm.AcquireGenericServiceTicket(ctx, &spec)
	if err != nil {
		return nil, nil, err
	}

	cookie := &http.Cookie{
		Name:  "vmware_cgi_ticket",
		Value: ticket.Id,
	}

	if d.useServiceTicketHostName(ticket.HostName) {
		u.Host = ticket.HostName
	}

	d.Client().SetThumbprint(u.Host, ticket.SslThumbprint)

	return u, cookie, nil
}

func (d Datastore) uploadTicket(ctx context.Context, path string, param *soap.Upload) (*url.URL, *soap.Upload, error) {
	p := soap.DefaultUpload
	if param != nil {
		p = *param // copy
	}

	u, ticket, err := d.ServiceTicket(ctx, path, p.Method)
	if err != nil {
		return nil, nil, err
	}

	p.Ticket = ticket

	return u, &p, nil
}

func (d Datastore) downloadTicket(ctx context.Context, path string, param *soap.Download) (*url.URL, *soap.Download, error) {
	p := soap.DefaultDownload
	if param != nil {
		p = *param // copy
	}

	u, ticket, err := d.ServiceTicket(ctx, path, p.Method)
	if err != nil {
		return nil, nil, err
	}

	p.Ticket = ticket

	return u, &p, nil
}

// Upload via soap.Upload with an http service ticket
func (d Datastore) Upload(ctx context.Context, f io.Reader, path string, param *soap.Upload) error {
	u, p, err := d.uploadTicket(ctx, path, param)
	if err != nil {
		return err
	}
	return d.Client().Upload(ctx, f, u, p)
}

// UploadFile via soap.Upload with an http service ticket
func (d Datastore) UploadFile(ctx context.Context, file string, path string, param *soap.Upload) error {
	u, p, err := d.uploadTicket(ctx, path, param)
	if err != nil {
		return err
	}
	return d.Client().UploadFile(ctx, file, u, p)
}

// Download via soap.Download with an http service ticket
func (d Datastore) Download(ctx context.Context, path string, param *soap.Download) (io.ReadCloser, int64, error) {
	u, p, err := d.downloadTicket(ctx, path, param)
	if err != nil {
		return nil, 0, err
	}
	return d.Client().Download(ctx, u, p)
}

// DownloadFile via soap.Download with an http service ticket
func (d Datastore) DownloadFile(ctx context.Context, path string, file string, param *soap.Download) error {
	u, p, err := d.downloadTicket(ctx, path, param)
	if err != nil {
		return err
	}
	return d.Client().DownloadFile(ctx, file, u, p)
}

// AttachedHosts returns hosts that have this Datastore attached, accessible and writable.
func (d Datastore) AttachedHosts(ctx context.Context) ([]*HostSystem, error) {
	var ds mo.Datastore
	var hosts []*HostSystem

	pc := property.DefaultCollector(d.Client())
	err := pc.RetrieveOne(ctx, d.Reference(), []string{"host"}, &ds)
	if err != nil {
		return nil, err
	}

	mounts := make(map[types.ManagedObjectReference]types.DatastoreHostMount)
	var refs []types.ManagedObjectReference
	for _, host := range ds.Host {
		refs = append(refs, host.Key)
		mounts[host.Key] = host
	}

	var hs []mo.HostSystem
	err = pc.Retrieve(ctx, refs, []string{"runtime.connectionState", "runtime.powerState"}, &hs)
	if err != nil {
		return nil, err
	}

	for _, host := range hs {
		if host.Runtime.ConnectionState == types.HostSystemConnectionStateConnected &&
			host.Runtime.PowerState == types.HostSystemPowerStatePoweredOn {

			mount := mounts[host.Reference()]
			info := mount.MountInfo

			if *info.Mounted && *info.Accessible && info.AccessMode == string(types.HostMountModeReadWrite) {
				hosts = append(hosts, NewHostSystem(d.Client(), mount.Key))
			}
		}
	}

	return hosts, nil
}

// AttachedClusterHosts returns hosts that have this Datastore attached, accessible and writable and are members of the given cluster.
func (d Datastore) AttachedClusterHosts(ctx context.Context, cluster *ComputeResource) ([]*HostSystem, error) {
	var hosts []*HostSystem

	clusterHosts, err := cluster.Hosts(ctx)
	if err != nil {
		return nil, err
	}

	attachedHosts, err := d.AttachedHosts(ctx)
	if err != nil {
		return nil, err
	}

	refs := make(map[types.ManagedObjectReference]bool)
	for _, host := range attachedHosts {
		refs[host.Reference()] = true
	}

	for _, host := range clusterHosts {
		if refs[host.Reference()] {
			hosts = append(hosts, host)
		}
	}

	return hosts, nil
}

func (d Datastore) Stat(ctx context.Context, file string) (types.BaseFileInfo, error) {
	b, err := d.Browser(ctx)
	if err != nil {
		return nil, err
	}

	spec := types.HostDatastoreBrowserSearchSpec{
		Details: &types.FileQueryFlags{
			FileType:     true,
			FileSize:     true,
			Modification: true,
			FileOwner:    types.NewBool(true),
		},
		MatchPattern: []string{path.Base(file)},
	}

	dsPath := d.Path(path.Dir(file))
	task, err := b.SearchDatastore(ctx, dsPath, &spec)
	if err != nil {
		return nil, err
	}

	info, err := task.WaitForResult(ctx, nil)
	if err != nil {
		if types.IsFileNotFound(err) {
			// FileNotFound means the base path doesn't exist.
			return nil, DatastoreNoSuchDirectoryError{"stat", dsPath}
		}

		return nil, err
	}

	res := info.Result.(types.HostDatastoreBrowserSearchResults)
	if len(res.File) == 0 {
		// File doesn't exist
		return nil, DatastoreNoSuchFileError{"stat", d.Path(file)}
	}

	return res.File[0], nil

}

// Type returns the type of file system volume.
func (d Datastore) Type(ctx context.Context) (types.HostFileSystemVolumeFileSystemType, error) {
	var mds mo.Datastore

	if err := d.Properties(ctx, d.Reference(), []string{"summary.type"}, &mds); err != nil {
		return types.HostFileSystemVolumeFileSystemType(""), err
	}
	return types.HostFileSystemVolumeFileSystemType(mds.Summary.Type), nil
}
