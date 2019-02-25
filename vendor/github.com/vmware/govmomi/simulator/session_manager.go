/*
Copyright (c) 2017-2018 VMware, Inc. All Rights Reserved.

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

package simulator

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/session"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

type SessionManager struct {
	mo.SessionManager

	ServiceHostName string

	sessions map[string]Session
}

func NewSessionManager(ref types.ManagedObjectReference) object.Reference {
	s := &SessionManager{
		sessions: make(map[string]Session),
	}
	s.Self = ref
	return s
}

func createSession(ctx *Context, name string, locale string) types.UserSession {
	now := time.Now().UTC()

	if locale == "" {
		locale = session.Locale
	}

	session := Session{
		UserSession: types.UserSession{
			Key:            uuid.New().String(),
			UserName:       name,
			FullName:       name,
			LoginTime:      now,
			LastActiveTime: now,
			Locale:         locale,
			MessageLocale:  locale,
		},
		Registry: NewRegistry(),
	}

	ctx.SetSession(session, true)

	return session.UserSession
}

func (s *SessionManager) Login(ctx *Context, req *types.Login) soap.HasFault {
	body := new(methods.LoginBody)

	if req.UserName == "" || req.Password == "" || ctx.Session != nil {
		body.Fault_ = invalidLogin
	} else {
		body.Res = &types.LoginResponse{
			Returnval: createSession(ctx, req.UserName, req.Locale),
		}
	}

	return body
}

func (s *SessionManager) LoginExtensionByCertificate(ctx *Context, req *types.LoginExtensionByCertificate) soap.HasFault {
	body := new(methods.LoginExtensionByCertificateBody)

	if ctx.req.TLS == nil || len(ctx.req.TLS.PeerCertificates) == 0 {
		body.Fault_ = Fault("", new(types.NoClientCertificate))
		return body
	}

	if req.ExtensionKey == "" || ctx.Session != nil {
		body.Fault_ = invalidLogin
	} else {
		body.Res = &types.LoginExtensionByCertificateResponse{
			Returnval: createSession(ctx, req.ExtensionKey, req.Locale),
		}
	}

	return body
}

func (s *SessionManager) LoginByToken(ctx *Context, req *types.LoginByToken) soap.HasFault {
	body := new(methods.LoginByTokenBody)

	if ctx.Session != nil {
		body.Fault_ = invalidLogin
	} else {
		var subject struct {
			ID string `xml:"Assertion>Subject>NameID"`
		}

		if s, ok := ctx.Header.Security.(*Element); ok {
			_ = s.Decode(&subject)
		}

		if subject.ID == "" {
			body.Fault_ = invalidLogin
			return body
		}

		body.Res = &types.LoginByTokenResponse{
			Returnval: createSession(ctx, subject.ID, req.Locale),
		}
	}

	return body
}

func (s *SessionManager) Logout(ctx *Context, _ *types.Logout) soap.HasFault {
	session := ctx.Session
	delete(s.sessions, session.Key)
	pc := Map.content().PropertyCollector

	for ref, obj := range ctx.Session.Registry.objects {
		if ref == pc {
			continue // don't unregister the PropertyCollector singleton
		}
		if _, ok := obj.(RegisterObject); ok {
			ctx.Map.Remove(ref) // Remove RegisterObject handlers
		}
	}

	ctx.postEvent(&types.UserLogoutSessionEvent{
		IpAddress: session.IpAddress,
		UserAgent: session.UserAgent,
		SessionId: session.Key,
		LoginTime: &session.LoginTime,
	})

	return &methods.LogoutBody{Res: new(types.LogoutResponse)}
}

func (s *SessionManager) TerminateSession(ctx *Context, req *types.TerminateSession) soap.HasFault {
	body := new(methods.TerminateSessionBody)

	for _, id := range req.SessionId {
		if id == ctx.Session.Key {
			body.Fault_ = Fault("", new(types.InvalidArgument))
			return body
		}
		delete(s.sessions, id)
	}

	body.Res = new(types.TerminateSessionResponse)
	return body
}

func (s *SessionManager) AcquireCloneTicket(ctx *Context, _ *types.AcquireCloneTicket) soap.HasFault {
	session := *ctx.Session
	session.Key = uuid.New().String()
	s.sessions[session.Key] = session

	return &methods.AcquireCloneTicketBody{
		Res: &types.AcquireCloneTicketResponse{
			Returnval: session.Key,
		},
	}
}

func (s *SessionManager) CloneSession(ctx *Context, ticket *types.CloneSession) soap.HasFault {
	body := new(methods.CloneSessionBody)

	session, exists := s.sessions[ticket.CloneTicket]

	if exists {
		delete(s.sessions, ticket.CloneTicket) // A clone ticket can only be used once
		session.Key = uuid.New().String()
		ctx.SetSession(session, true)

		body.Res = &types.CloneSessionResponse{
			Returnval: session.UserSession,
		}
	} else {
		body.Fault_ = invalidLogin
	}

	return body
}

func (s *SessionManager) AcquireGenericServiceTicket(ticket *types.AcquireGenericServiceTicket) soap.HasFault {
	return &methods.AcquireGenericServiceTicketBody{
		Res: &types.AcquireGenericServiceTicketResponse{
			Returnval: types.SessionManagerGenericServiceTicket{
				Id:       uuid.New().String(),
				HostName: s.ServiceHostName,
			},
		},
	}
}

// internalContext is the session for use by the in-memory client (Service.RoundTrip)
var internalContext = &Context{
	Context: context.Background(),
	Session: &Session{
		UserSession: types.UserSession{
			Key: uuid.New().String(),
		},
		Registry: NewRegistry(),
	},
	Map: Map,
}

var invalidLogin = Fault("Login failure", new(types.InvalidLogin))

// Context provides per-request Session management.
type Context struct {
	req *http.Request
	res http.ResponseWriter
	m   *SessionManager

	context.Context
	Session *Session
	Header  soap.Header
	Caller  *types.ManagedObjectReference
	Map     *Registry
}

// mapSession maps an HTTP cookie to a Session.
func (c *Context) mapSession() {
	if cookie, err := c.req.Cookie(soap.SessionCookieName); err == nil {
		if val, ok := c.m.sessions[cookie.Value]; ok {
			c.SetSession(val, false)
		}
	}
}

// SetSession should be called after successful authentication.
func (c *Context) SetSession(session Session, login bool) {
	session.UserAgent = c.req.UserAgent()
	session.IpAddress = strings.Split(c.req.RemoteAddr, ":")[0]
	session.LastActiveTime = time.Now()

	c.m.sessions[session.Key] = session
	c.Session = &session

	if login {
		http.SetCookie(c.res, &http.Cookie{
			Name:  soap.SessionCookieName,
			Value: session.Key,
		})

		c.postEvent(&types.UserLoginSessionEvent{
			SessionId: session.Key,
			IpAddress: session.IpAddress,
			UserAgent: session.UserAgent,
			Locale:    session.Locale,
		})
	}
}

// WithLock holds a lock for the given object while then given function is run.
func (c *Context) WithLock(obj mo.Reference, f func()) {
	if c.Caller != nil && *c.Caller == obj.Reference() {
		// Internal method invocation, obj is already locked
		f()
		return
	}
	Map.WithLock(obj, f)
}

// postEvent wraps EventManager.PostEvent for internal use, with a lock on the EventManager.
func (c *Context) postEvent(events ...types.BaseEvent) {
	m := Map.EventManager()
	c.WithLock(m, func() {
		for _, event := range events {
			m.PostEvent(c, &types.PostEvent{EventToPost: event})
		}
	})
}

// Session combines a UserSession and a Registry for per-session managed objects.
type Session struct {
	types.UserSession
	*Registry
}

// Put wraps Registry.Put, setting the moref value to include the session key.
func (s *Session) Put(item mo.Reference) mo.Reference {
	ref := item.Reference()
	if ref.Value == "" {
		ref.Value = fmt.Sprintf("session[%s]%s", s.Key, uuid.New())
	}
	s.Registry.setReference(item, ref)
	return s.Registry.Put(item)
}

// Get wraps Registry.Get, session-izing singleton objects such as SessionManager and the root PropertyCollector.
func (s *Session) Get(ref types.ManagedObjectReference) mo.Reference {
	obj := s.Registry.Get(ref)
	if obj != nil {
		return obj
	}

	// Return a session "view" of certain singleton objects
	switch ref.Type {
	case "SessionManager":
		// Clone SessionManager so the PropertyCollector can properly report CurrentSession
		m := *Map.SessionManager()
		m.CurrentSession = &s.UserSession

		// TODO: we could maintain SessionList as part of the SessionManager singleton
		for _, session := range m.sessions {
			m.SessionList = append(m.SessionList, session.UserSession)
		}

		return &m
	case "PropertyCollector":
		if ref == Map.content().PropertyCollector {
			return s.Put(NewPropertyCollector(ref))
		}
	}

	return Map.Get(ref)
}
