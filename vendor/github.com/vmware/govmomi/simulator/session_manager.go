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
	"os"
	"reflect"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"

	"github.com/vmware/govmomi/session"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

type SessionManager struct {
	mo.SessionManager
	nopLocker

	ServiceHostName string
	TLSCert         func() string

	sessions map[string]Session
}

func (m *SessionManager) init(*Registry) {
	m.sessions = make(map[string]Session)
}

var (
	// SessionIdleTimeout duration used to expire idle sessions
	SessionIdleTimeout time.Duration

	sessionMutex sync.Mutex

	// secureCookies enables Set-Cookie.Secure=true
	// We can't do this by default as simulator.Service defaults to no TLS by default and
	// Go's cookiejar does not send Secure cookies unless the URL scheme is https.
	secureCookies = os.Getenv("VCSIM_SECURE_COOKIES") == "true"
)

func createSession(ctx *Context, name string, locale string) types.UserSession {
	now := time.Now().UTC()

	if locale == "" {
		locale = session.Locale
	}

	session := Session{
		UserSession: types.UserSession{
			Key:              uuid.New().String(),
			UserName:         name,
			FullName:         name,
			LoginTime:        now,
			LastActiveTime:   now,
			Locale:           locale,
			MessageLocale:    locale,
			ExtensionSession: types.NewBool(false),
		},
		Registry: NewRegistry(),
	}

	ctx.SetSession(session, true)

	return ctx.Session.UserSession
}

func (m *SessionManager) getSession(id string) (Session, bool) {
	sessionMutex.Lock()
	defer sessionMutex.Unlock()
	s, ok := m.sessions[id]
	return s, ok
}

func (m *SessionManager) delSession(id string) {
	sessionMutex.Lock()
	defer sessionMutex.Unlock()
	delete(m.sessions, id)
}

func (m *SessionManager) putSession(s Session) {
	sessionMutex.Lock()
	defer sessionMutex.Unlock()
	m.sessions[s.Key] = s
}

func (s *SessionManager) validLogin(ctx *Context, req *types.Login) bool {
	if ctx.Session != nil {
		return false
	}
	user := ctx.svc.Listen.User
	if user == nil || user == DefaultLogin {
		return req.UserName != "" && req.Password != ""
	}
	pass, _ := user.Password()
	return req.UserName == user.Username() && req.Password == pass
}

func (s *SessionManager) Login(ctx *Context, req *types.Login) soap.HasFault {
	body := new(methods.LoginBody)

	if s.validLogin(ctx, req) {
		body.Res = &types.LoginResponse{
			Returnval: createSession(ctx, req.UserName, req.Locale),
		}
	} else {
		body.Fault_ = invalidLogin
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
	s.delSession(session.Key)
	pc := ctx.Map.content().PropertyCollector

	for ref, obj := range ctx.Session.Registry.objects {
		if ref == pc {
			continue // don't unregister the PropertyCollector singleton
		}
		if _, ok := obj.(RegisterObject); ok {
			ctx.Map.Remove(ctx, ref) // Remove RegisterObject handlers
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
		if _, ok := s.getSession(id); !ok {
			body.Fault_ = Fault("", new(types.NotFound))
			return body
		}
		s.delSession(id)
	}

	body.Res = new(types.TerminateSessionResponse)
	return body
}

func (s *SessionManager) SessionIsActive(ctx *Context, req *types.SessionIsActive) soap.HasFault {
	body := new(methods.SessionIsActiveBody)

	if ctx.Map.IsESX() {
		body.Fault_ = Fault("", new(types.NotImplemented))
		return body
	}

	body.Res = new(types.SessionIsActiveResponse)

	if session, exists := s.getSession(req.SessionID); exists {
		body.Res.Returnval = session.UserName == req.UserName
	}

	return body
}

func (s *SessionManager) AcquireCloneTicket(ctx *Context, _ *types.AcquireCloneTicket) soap.HasFault {
	session := *ctx.Session
	session.Key = uuid.New().String()
	s.putSession(session)

	return &methods.AcquireCloneTicketBody{
		Res: &types.AcquireCloneTicketResponse{
			Returnval: session.Key,
		},
	}
}

func (s *SessionManager) CloneSession(ctx *Context, ticket *types.CloneSession) soap.HasFault {
	body := new(methods.CloneSessionBody)

	session, exists := s.getSession(ticket.CloneTicket)

	if exists {
		s.delSession(ticket.CloneTicket) // A clone ticket can only be used once
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

var invalidLogin = Fault("Login failure", new(types.InvalidLogin))

// Context provides per-request Session management.
type Context struct {
	req *http.Request
	res http.ResponseWriter
	svc *Service

	context.Context
	Session *Session
	Header  soap.Header
	Caller  *types.ManagedObjectReference
	Map     *Registry
}

// mapSession maps an HTTP cookie to a Session.
func (c *Context) mapSession() {
	if cookie, err := c.req.Cookie(soap.SessionCookieName); err == nil {
		if val, ok := c.svc.sm.getSession(cookie.Value); ok {
			c.SetSession(val, false)
		}
	}
}

func (m *SessionManager) expiredSession(id string, now time.Time) bool {
	expired := true

	s, ok := m.getSession(id)
	if ok {
		expired = now.Sub(s.LastActiveTime) > SessionIdleTimeout
		if expired {
			m.delSession(id)
		}
	}

	return expired
}

// SessionIdleWatch starts a goroutine that calls func expired() at SessionIdleTimeout intervals.
// The goroutine exits if the func returns true.
func SessionIdleWatch(ctx context.Context, id string, expired func(string, time.Time) bool) {
	if SessionIdleTimeout == 0 {
		return
	}

	go func() {
		for t := time.NewTimer(SessionIdleTimeout); ; {
			select {
			case <-ctx.Done():
				return
			case now := <-t.C:
				if expired(id, now) {
					return
				}
				t.Reset(SessionIdleTimeout)
			}
		}
	}()
}

// SetSession should be called after successful authentication.
func (c *Context) SetSession(session Session, login bool) {
	session.UserAgent = c.req.UserAgent()
	session.IpAddress = strings.Split(c.req.RemoteAddr, ":")[0]
	session.LastActiveTime = time.Now()
	session.CallCount++

	c.svc.sm.putSession(session)
	c.Session = &session

	if login {
		http.SetCookie(c.res, &http.Cookie{
			Name:     soap.SessionCookieName,
			Value:    session.Key,
			Secure:   secureCookies,
			HttpOnly: true,
		})

		c.postEvent(&types.UserLoginSessionEvent{
			SessionId: session.Key,
			IpAddress: session.IpAddress,
			UserAgent: session.UserAgent,
			Locale:    session.Locale,
		})

		SessionIdleWatch(c.Context, session.Key, c.svc.sm.expiredSession)
	}
}

// WithLock holds a lock for the given object while the given function is run.
// It will skip locking if this context already holds the given object's lock.
func (c *Context) WithLock(obj mo.Reference, f func()) {
	// TODO: This is not always going to be correct. An object should
	// really be locked by the registry that "owns it", which is not always
	// Map. This function will need to take the Registry as an additional
	// argument to accomplish this.
	// Basic mutex locking will work even if obj doesn't belong to Map, but
	// if obj implements sync.Locker, that custom locking will not be used.
	c.Map.WithLock(c, obj, f)
}

// postEvent wraps EventManager.PostEvent for internal use, with a lock on the EventManager.
func (c *Context) postEvent(events ...types.BaseEvent) {
	m := c.Map.EventManager()
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

func (s *Session) setReference(item mo.Reference) {
	ref := item.Reference()
	if ref.Value == "" {
		ref.Value = fmt.Sprintf("session[%s]%s", s.Key, uuid.New())
	}
	if ref.Type == "" {
		ref.Type = typeName(item)
	}
	s.Registry.setReference(item, ref)
}

// Put wraps Registry.Put, setting the moref value to include the session key.
func (s *Session) Put(item mo.Reference) mo.Reference {
	s.setReference(item)
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
		sessionMutex.Lock()
		for _, session := range m.sessions {
			m.SessionList = append(m.SessionList, session.UserSession)
		}
		sessionMutex.Unlock()

		return &m
	case "PropertyCollector":
		if ref == Map.content().PropertyCollector {
			// Per-session instance of the PropertyCollector singleton.
			// Using reflection here as PropertyCollector might be wrapped with a custom type.
			obj = Map.Get(ref)
			pc := reflect.New(reflect.TypeOf(obj).Elem())
			obj = pc.Interface().(mo.Reference)
			s.Registry.setReference(obj, ref)
			return s.Put(obj)
		}
	}

	return Map.Get(ref)
}
