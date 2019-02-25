/*
Copyright (c) 2015 VMware, Inc. All Rights Reserved.

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

package session

import (
	"context"
	"net/http"
	"net/url"
	"os"

	"github.com/vmware/govmomi/property"
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
)

// Locale defaults to "en_US" and can be overridden via this var or the GOVMOMI_LOCALE env var.
// A value of "_" uses the server locale setting.
var Locale = os.Getenv("GOVMOMI_LOCALE")

func init() {
	if Locale == "_" {
		Locale = ""
	} else if Locale == "" {
		Locale = "en_US"
	}
}

type Manager struct {
	client      *vim25.Client
	userSession *types.UserSession
}

func NewManager(client *vim25.Client) *Manager {
	m := Manager{
		client: client,
	}

	return &m
}

func (sm Manager) Reference() types.ManagedObjectReference {
	return *sm.client.ServiceContent.SessionManager
}

func (sm *Manager) SetLocale(ctx context.Context, locale string) error {
	req := types.SetLocale{
		This:   sm.Reference(),
		Locale: locale,
	}

	_, err := methods.SetLocale(ctx, sm.client, &req)
	return err
}

func (sm *Manager) Login(ctx context.Context, u *url.Userinfo) error {
	req := types.Login{
		This:   sm.Reference(),
		Locale: Locale,
	}

	if u != nil {
		req.UserName = u.Username()
		if pw, ok := u.Password(); ok {
			req.Password = pw
		}
	}

	login, err := methods.Login(ctx, sm.client, &req)
	if err != nil {
		return err
	}

	sm.userSession = &login.Returnval
	return nil
}

// LoginExtensionByCertificate uses the vCenter SDK tunnel to login using a client certificate.
// The client certificate can be set using the soap.Client.SetCertificate method.
// See: https://kb.vmware.com/s/article/2004305
func (sm *Manager) LoginExtensionByCertificate(ctx context.Context, key string) error {
	c := sm.client
	u := c.URL()
	if u.Hostname() != "sdkTunnel" {
		sc := c.Tunnel()
		c = &vim25.Client{
			Client:         sc,
			RoundTripper:   sc,
			ServiceContent: c.ServiceContent,
		}
		// When http.Transport.Proxy is used, our thumbprint checker is bypassed, resulting in:
		// "Post https://sdkTunnel:8089/sdk: x509: certificate is valid for $vcenter_hostname, not sdkTunnel"
		// The only easy way around this is to disable verification for the call to LoginExtensionByCertificate().
		// TODO: find a way to avoid disabling InsecureSkipVerify.
		c.Transport.(*http.Transport).TLSClientConfig.InsecureSkipVerify = true
	}

	req := types.LoginExtensionByCertificate{
		This:         sm.Reference(),
		ExtensionKey: key,
		Locale:       Locale,
	}

	login, err := methods.LoginExtensionByCertificate(ctx, c, &req)
	if err != nil {
		return err
	}

	// Copy the session cookie
	sm.client.Jar.SetCookies(u, c.Jar.Cookies(c.URL()))

	sm.userSession = &login.Returnval
	return nil
}

func (sm *Manager) LoginByToken(ctx context.Context) error {
	req := types.LoginByToken{
		This:   sm.Reference(),
		Locale: Locale,
	}

	login, err := methods.LoginByToken(ctx, sm.client, &req)
	if err != nil {
		return err
	}

	sm.userSession = &login.Returnval
	return nil
}

func (sm *Manager) Logout(ctx context.Context) error {
	req := types.Logout{
		This: sm.Reference(),
	}

	_, err := methods.Logout(ctx, sm.client, &req)
	if err != nil {
		return err
	}

	sm.userSession = nil
	return nil
}

// UserSession retrieves and returns the SessionManager's CurrentSession field.
// Nil is returned if the session is not authenticated.
func (sm *Manager) UserSession(ctx context.Context) (*types.UserSession, error) {
	var mgr mo.SessionManager

	pc := property.DefaultCollector(sm.client)
	err := pc.RetrieveOne(ctx, sm.Reference(), []string{"currentSession"}, &mgr)
	if err != nil {
		// It's OK if we can't retrieve properties because we're not authenticated
		if f, ok := err.(types.HasFault); ok {
			switch f.Fault().(type) {
			case *types.NotAuthenticated:
				return nil, nil
			}
		}

		return nil, err
	}

	return mgr.CurrentSession, nil
}

func (sm *Manager) TerminateSession(ctx context.Context, sessionId []string) error {
	req := types.TerminateSession{
		This:      sm.Reference(),
		SessionId: sessionId,
	}

	_, err := methods.TerminateSession(ctx, sm.client, &req)
	return err
}

// SessionIsActive checks whether the session that was created at login is
// still valid. This function only works against vCenter.
func (sm *Manager) SessionIsActive(ctx context.Context) (bool, error) {
	if sm.userSession == nil {
		return false, nil
	}

	req := types.SessionIsActive{
		This:      sm.Reference(),
		SessionID: sm.userSession.Key,
		UserName:  sm.userSession.UserName,
	}

	active, err := methods.SessionIsActive(ctx, sm.client, &req)
	if err != nil {
		return false, err
	}

	return active.Returnval, err
}

func (sm *Manager) AcquireGenericServiceTicket(ctx context.Context, spec types.BaseSessionManagerServiceRequestSpec) (*types.SessionManagerGenericServiceTicket, error) {
	req := types.AcquireGenericServiceTicket{
		This: sm.Reference(),
		Spec: spec,
	}

	res, err := methods.AcquireGenericServiceTicket(ctx, sm.client, &req)
	if err != nil {
		return nil, err
	}

	return &res.Returnval, nil
}

func (sm *Manager) AcquireLocalTicket(ctx context.Context, userName string) (*types.SessionManagerLocalTicket, error) {
	req := types.AcquireLocalTicket{
		This:     sm.Reference(),
		UserName: userName,
	}

	res, err := methods.AcquireLocalTicket(ctx, sm.client, &req)
	if err != nil {
		return nil, err
	}

	return &res.Returnval, nil
}

func (sm *Manager) AcquireCloneTicket(ctx context.Context) (string, error) {
	req := types.AcquireCloneTicket{
		This: sm.Reference(),
	}

	res, err := methods.AcquireCloneTicket(ctx, sm.client, &req)
	if err != nil {
		return "", err
	}

	return res.Returnval, nil
}

func (sm *Manager) CloneSession(ctx context.Context, ticket string) error {
	req := types.CloneSession{
		This:        sm.Reference(),
		CloneTicket: ticket,
	}

	res, err := methods.CloneSession(ctx, sm.client, &req)
	if err != nil {
		return err
	}

	sm.userSession = &res.Returnval
	return nil
}
