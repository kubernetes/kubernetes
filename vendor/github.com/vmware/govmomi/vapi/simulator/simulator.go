/*
Copyright (c) 2018-2023 VMware, Inc. All Rights Reserved.

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
	"archive/tar"
	"bytes"
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"

	"github.com/vmware/govmomi"
	"github.com/vmware/govmomi/nfc"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/ovf"
	"github.com/vmware/govmomi/simulator"
	"github.com/vmware/govmomi/vapi"
	"github.com/vmware/govmomi/vapi/internal"
	"github.com/vmware/govmomi/vapi/library"
	"github.com/vmware/govmomi/vapi/rest"
	"github.com/vmware/govmomi/vapi/tags"
	"github.com/vmware/govmomi/vapi/vcenter"
	"github.com/vmware/govmomi/view"
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/types"
	vim "github.com/vmware/govmomi/vim25/types"
	"github.com/vmware/govmomi/vim25/xml"
)

type item struct {
	*library.Item
	File     []library.File
	Template *types.ManagedObjectReference
}

type content struct {
	*library.Library
	Item map[string]*item
	Subs map[string]*library.Subscriber
	VMTX map[string]*types.ManagedObjectReference
}

type update struct {
	*library.Session
	Library *library.Library
	File    map[string]*library.UpdateFile
}

type download struct {
	*library.Session
	Library *library.Library
	File    map[string]*library.DownloadFile
}

type handler struct {
	sync.Mutex
	sm          *simulator.SessionManager
	ServeMux    *http.ServeMux
	URL         url.URL
	Category    map[string]*tags.Category
	Tag         map[string]*tags.Tag
	Association map[string]map[internal.AssociatedObject]bool
	Session     map[string]*rest.Session
	Library     map[string]*content
	Update      map[string]update
	Download    map[string]download
	Policies    []library.ContentSecurityPoliciesInfo
	Trust       map[string]library.TrustedCertificate
}

func init() {
	simulator.RegisterEndpoint(func(s *simulator.Service, r *simulator.Registry) {
		if r.IsVPX() {
			patterns, h := New(s.Listen, r)
			for _, p := range patterns {
				s.Handle(p, h)
			}
		}
	})
}

// New creates a vAPI simulator.
func New(u *url.URL, r *simulator.Registry) ([]string, http.Handler) {
	s := &handler{
		sm:          r.SessionManager(),
		ServeMux:    http.NewServeMux(),
		URL:         *u,
		Category:    make(map[string]*tags.Category),
		Tag:         make(map[string]*tags.Tag),
		Association: make(map[string]map[internal.AssociatedObject]bool),
		Session:     make(map[string]*rest.Session),
		Library:     make(map[string]*content),
		Update:      make(map[string]update),
		Download:    make(map[string]download),
		Policies:    defaultSecurityPolicies(),
		Trust:       make(map[string]library.TrustedCertificate),
	}

	handlers := []struct {
		p string
		m http.HandlerFunc
	}{
		// /rest/ patterns.
		{internal.SessionPath, s.session},
		{internal.CategoryPath, s.category},
		{internal.CategoryPath + "/", s.categoryID},
		{internal.TagPath, s.tag},
		{internal.TagPath + "/", s.tagID},
		{internal.AssociationPath, s.association},
		{internal.AssociationPath + "/", s.associationID},
		{internal.LibraryPath, s.library},
		{internal.LocalLibraryPath, s.library},
		{internal.SubscribedLibraryPath, s.library},
		{internal.LibraryPath + "/", s.libraryID},
		{internal.LocalLibraryPath + "/", s.libraryID},
		{internal.SubscribedLibraryPath + "/", s.libraryID},
		{internal.Subscriptions, s.subscriptions},
		{internal.Subscriptions + "/", s.subscriptionsID},
		{internal.LibraryItemPath, s.libraryItem},
		{internal.LibraryItemPath + "/", s.libraryItemID},
		{internal.SubscribedLibraryItem + "/", s.libraryItemID},
		{internal.LibraryItemUpdateSession, s.libraryItemUpdateSession},
		{internal.LibraryItemUpdateSession + "/", s.libraryItemUpdateSessionID},
		{internal.LibraryItemUpdateSessionFile, s.libraryItemUpdateSessionFile},
		{internal.LibraryItemUpdateSessionFile + "/", s.libraryItemUpdateSessionFileID},
		{internal.LibraryItemDownloadSession, s.libraryItemDownloadSession},
		{internal.LibraryItemDownloadSession + "/", s.libraryItemDownloadSessionID},
		{internal.LibraryItemDownloadSessionFile, s.libraryItemDownloadSessionFile},
		{internal.LibraryItemDownloadSessionFile + "/", s.libraryItemDownloadSessionFileID},
		{internal.LibraryItemFileData + "/", s.libraryItemFileData},
		{internal.LibraryItemFilePath, s.libraryItemFile},
		{internal.LibraryItemFilePath + "/", s.libraryItemFileID},
		{internal.VCenterOVFLibraryItem, s.libraryItemOVF},
		{internal.VCenterOVFLibraryItem + "/", s.libraryItemOVFID},
		{internal.VCenterVMTXLibraryItem, s.libraryItemCreateTemplate},
		{internal.VCenterVMTXLibraryItem + "/", s.libraryItemTemplateID},
		{internal.VCenterVM + "/", s.vmID},
		{internal.DebugEcho, s.debugEcho},
		// /api/ patterns.
		{internal.SecurityPoliciesPath, s.librarySecurityPolicies},
		{internal.TrustedCertificatesPath, s.libraryTrustedCertificates},
		{internal.TrustedCertificatesPath + "/", s.libraryTrustedCertificatesID},
	}

	for i := range handlers {
		h := handlers[i]
		s.HandleFunc(h.p, h.m)
	}

	return []string{rest.Path + "/", vapi.Path + "/"}, s
}

func (s *handler) withClient(f func(context.Context, *vim25.Client) error) error {
	ctx := context.Background()
	c, err := govmomi.NewClient(ctx, &s.URL, true)
	if err != nil {
		return err
	}
	defer func() {
		_ = c.Logout(ctx)
	}()
	return f(ctx, c.Client)
}

// HandleFunc wraps the given handler with authorization checks and passes to http.ServeMux.HandleFunc
func (s *handler) HandleFunc(pattern string, handler func(http.ResponseWriter, *http.Request)) {
	// Rest paths have been moved from /rest/* to /api/*. Account for both the legacy and new cases here.
	if !strings.HasPrefix(pattern, rest.Path) && !strings.HasPrefix(pattern, vapi.Path) {
		pattern = rest.Path + pattern
	}

	s.ServeMux.HandleFunc(pattern, func(w http.ResponseWriter, r *http.Request) {
		s.Lock()
		defer s.Unlock()

		if !s.isAuthorized(r) {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}

		handler(w, r)
	})
}

func (s *handler) isAuthorized(r *http.Request) bool {
	if r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, internal.SessionPath) && s.action(r) == "" {
		return true
	}
	id := r.Header.Get(internal.SessionCookieName)
	if id == "" {
		if cookie, err := r.Cookie(internal.SessionCookieName); err == nil {
			id = cookie.Value
			r.Header.Set(internal.SessionCookieName, id)
		}
	}
	info, ok := s.Session[id]
	if ok {
		info.LastAccessed = time.Now()
	} else {
		_, ok = s.Update[id]
	}
	return ok
}

func (s *handler) hasAuthorization(r *http.Request) (string, bool) {
	u, p, ok := r.BasicAuth()
	if ok { // user+pass auth
		return u, s.sm.Authenticate(s.URL, &vim.Login{UserName: u, Password: p})
	}
	auth := r.Header.Get("Authorization")
	return "TODO", strings.HasPrefix(auth, "SIGN ") // token auth
}

func (s *handler) findTag(e vim.VslmTagEntry) *tags.Tag {
	for _, c := range s.Category {
		if c.Name == e.ParentCategoryName {
			for _, t := range s.Tag {
				if t.Name == e.TagName && t.CategoryID == c.ID {
					return t
				}
			}
		}
	}
	return nil
}

// AttachedObjects is meant for internal use via simulator.Registry.tagManager
func (s *handler) AttachedObjects(tag vim.VslmTagEntry) ([]vim.ManagedObjectReference, vim.BaseMethodFault) {
	t := s.findTag(tag)
	if t == nil {
		return nil, new(vim.NotFound)
	}
	var ids []vim.ManagedObjectReference
	for id := range s.Association[t.ID] {
		ids = append(ids, vim.ManagedObjectReference(id))
	}
	return ids, nil
}

// AttachedTags is meant for internal use via simulator.Registry.tagManager
func (s *handler) AttachedTags(ref vim.ManagedObjectReference) ([]vim.VslmTagEntry, vim.BaseMethodFault) {
	oid := internal.AssociatedObject(ref)
	var tags []vim.VslmTagEntry
	for id, objs := range s.Association {
		if objs[oid] {
			tag := s.Tag[id]
			cat := s.Category[tag.CategoryID]
			tags = append(tags, vim.VslmTagEntry{
				TagName:            tag.Name,
				ParentCategoryName: cat.Name,
			})
		}
	}
	return tags, nil
}

// AttachTag is meant for internal use via simulator.Registry.tagManager
func (s *handler) AttachTag(ref vim.ManagedObjectReference, tag vim.VslmTagEntry) vim.BaseMethodFault {
	t := s.findTag(tag)
	if t == nil {
		return new(vim.NotFound)
	}
	s.Association[t.ID][internal.AssociatedObject(ref)] = true
	return nil
}

// DetachTag is meant for internal use via simulator.Registry.tagManager
func (s *handler) DetachTag(id vim.ManagedObjectReference, tag vim.VslmTagEntry) vim.BaseMethodFault {
	t := s.findTag(tag)
	if t == nil {
		return new(vim.NotFound)
	}
	delete(s.Association[t.ID], internal.AssociatedObject(id))
	return nil
}

// StatusOK responds with http.StatusOK and encodes val, if specified, to JSON
// For use with "/api" endpoints.
func StatusOK(w http.ResponseWriter, val ...interface{}) {
	w.WriteHeader(http.StatusOK)
	if len(val) == 0 {
		return
	}

	err := json.NewEncoder(w).Encode(val[0])

	if err != nil {
		log.Panic(err)
	}
}

// OK responds with http.StatusOK and encodes val, if specified, to JSON
// For use with "/rest" endpoints where the response is a "value" wrapped structure.
func OK(w http.ResponseWriter, val ...interface{}) {
	if len(val) == 0 {
		w.WriteHeader(http.StatusOK)
		return
	}

	s := struct {
		Value interface{} `json:"value,omitempty"`
	}{
		val[0],
	}

	StatusOK(w, s)
}

// BadRequest responds with http.StatusBadRequest and json encoded vAPI error of type kind.
// For use with "/rest" endpoints where the response is a "value" wrapped structure.
func BadRequest(w http.ResponseWriter, kind string) {
	w.WriteHeader(http.StatusBadRequest)

	err := json.NewEncoder(w).Encode(struct {
		Type  string `json:"type"`
		Value struct {
			Messages []string `json:"messages,omitempty"`
		} `json:"value,omitempty"`
	}{
		Type: kind,
	})

	if err != nil {
		log.Panic(err)
	}
}

func (*handler) error(w http.ResponseWriter, err error) {
	http.Error(w, err.Error(), http.StatusInternalServerError)
	log.Print(err)
}

// ServeHTTP handles vAPI requests.
func (s *handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodPost, http.MethodDelete, http.MethodGet, http.MethodPatch, http.MethodPut:
	default:
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	h, _ := s.ServeMux.Handler(r)
	h.ServeHTTP(w, r)
}

func (s *handler) decode(r *http.Request, w http.ResponseWriter, val interface{}) bool {
	return Decode(r, w, val)
}

// Decode the request Body into val.
// Returns true on success, otherwise false and sends the http.StatusBadRequest response.
func Decode(r *http.Request, w http.ResponseWriter, val interface{}) bool {
	defer r.Body.Close()
	err := json.NewDecoder(r.Body).Decode(val)
	if err != nil {
		log.Printf("%s %s: %s", r.Method, r.RequestURI, err)
		w.WriteHeader(http.StatusBadRequest)
		return false
	}
	return true
}

func (s *handler) expiredSession(id string, now time.Time) bool {
	expired := true
	s.Lock()
	session, ok := s.Session[id]
	if ok {
		expired = now.Sub(session.LastAccessed) > simulator.SessionIdleTimeout
		if expired {
			delete(s.Session, id)
		}
	}
	s.Unlock()
	return expired
}

func (s *handler) session(w http.ResponseWriter, r *http.Request) {
	id := r.Header.Get(internal.SessionCookieName)
	useHeaderAuthn := strings.ToLower(r.Header.Get(internal.UseHeaderAuthn))

	switch r.Method {
	case http.MethodPost:
		if s.action(r) != "" {
			if session, ok := s.Session[id]; ok {
				OK(w, session)
			} else {
				w.WriteHeader(http.StatusUnauthorized)
			}
			return
		}
		user, ok := s.hasAuthorization(r)
		if !ok {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		id = uuid.New().String()
		now := time.Now()
		s.Session[id] = &rest.Session{User: user, Created: now, LastAccessed: now}
		simulator.SessionIdleWatch(context.Background(), id, s.expiredSession)
		if useHeaderAuthn != "true" {
			http.SetCookie(w, &http.Cookie{
				Name:  internal.SessionCookieName,
				Value: id,
				Path:  rest.Path,
			})
		}
		OK(w, id)
	case http.MethodDelete:
		delete(s.Session, id)
		OK(w)
	case http.MethodGet:
		OK(w, s.Session[id])
	}
}

func (s *handler) action(r *http.Request) string {
	return r.URL.Query().Get("~action")
}

func (s *handler) id(r *http.Request) string {
	base := path.Base(r.URL.Path)
	id := strings.TrimPrefix(base, "id:")
	if id == base {
		return "" // trigger 404 Not Found w/o id: prefix
	}
	return id
}

func newID(kind string) string {
	return fmt.Sprintf("urn:vmomi:InventoryService%s:%s:GLOBAL", kind, uuid.New().String())
}

func (s *handler) category(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodPost:
		var spec struct {
			Category tags.Category `json:"create_spec"`
		}
		if s.decode(r, w, &spec) {
			for _, category := range s.Category {
				if category.Name == spec.Category.Name {
					BadRequest(w, "com.vmware.vapi.std.errors.already_exists")
					return
				}
			}
			id := newID("Category")
			spec.Category.ID = id
			s.Category[id] = &spec.Category
			OK(w, id)
		}
	case http.MethodGet:
		var ids []string
		for id := range s.Category {
			ids = append(ids, id)
		}

		OK(w, ids)
	}
}

func (s *handler) categoryID(w http.ResponseWriter, r *http.Request) {
	id := s.id(r)

	o, ok := s.Category[id]
	if !ok {
		http.NotFound(w, r)
		return
	}

	switch r.Method {
	case http.MethodDelete:
		delete(s.Category, id)
		for ix, tag := range s.Tag {
			if tag.CategoryID == id {
				delete(s.Tag, ix)
				delete(s.Association, ix)
			}
		}
		OK(w)
	case http.MethodPatch:
		var spec struct {
			Category tags.Category `json:"update_spec"`
		}
		if s.decode(r, w, &spec) {
			ntypes := len(spec.Category.AssociableTypes)
			if ntypes != 0 {
				// Validate that AssociableTypes is only appended to.
				etypes := len(o.AssociableTypes)
				fail := ntypes < etypes
				if !fail {
					fail = !reflect.DeepEqual(o.AssociableTypes, spec.Category.AssociableTypes[:etypes])
				}
				if fail {
					BadRequest(w, "com.vmware.vapi.std.errors.invalid_argument")
					return
				}
			}
			o.Patch(&spec.Category)
			OK(w)
		}
	case http.MethodGet:
		OK(w, o)
	}
}

func (s *handler) tag(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodPost:
		var spec struct {
			Tag tags.Tag `json:"create_spec"`
		}
		if s.decode(r, w, &spec) {
			for _, tag := range s.Tag {
				if tag.Name == spec.Tag.Name && tag.CategoryID == spec.Tag.CategoryID {
					BadRequest(w, "com.vmware.vapi.std.errors.already_exists")
					return
				}
			}
			id := newID("Tag")
			spec.Tag.ID = id
			s.Tag[id] = &spec.Tag
			s.Association[id] = make(map[internal.AssociatedObject]bool)
			OK(w, id)
		}
	case http.MethodGet:
		var ids []string
		for id := range s.Tag {
			ids = append(ids, id)
		}
		OK(w, ids)
	}
}

func (s *handler) tagID(w http.ResponseWriter, r *http.Request) {
	id := s.id(r)

	switch s.action(r) {
	case "list-tags-for-category":
		var ids []string
		for _, tag := range s.Tag {
			if tag.CategoryID == id {
				ids = append(ids, tag.ID)
			}
		}
		OK(w, ids)
		return
	}

	o, ok := s.Tag[id]
	if !ok {
		log.Printf("tag not found: %s", id)
		http.NotFound(w, r)
		return
	}

	switch r.Method {
	case http.MethodDelete:
		delete(s.Tag, id)
		delete(s.Association, id)
		OK(w)
	case http.MethodPatch:
		var spec struct {
			Tag tags.Tag `json:"update_spec"`
		}
		if s.decode(r, w, &spec) {
			o.Patch(&spec.Tag)
			OK(w)
		}
	case http.MethodGet:
		OK(w, o)
	}
}

// TODO: support cardinality checks
func (s *handler) association(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	var spec struct {
		internal.Association
		TagIDs    []string                    `json:"tag_ids,omitempty"`
		ObjectIDs []internal.AssociatedObject `json:"object_ids,omitempty"`
	}
	if !s.decode(r, w, &spec) {
		return
	}

	switch s.action(r) {
	case "list-attached-tags":
		var ids []string
		for id, objs := range s.Association {
			if objs[*spec.ObjectID] {
				ids = append(ids, id)
			}
		}
		OK(w, ids)

	case "list-attached-objects-on-tags":
		var res []tags.AttachedObjects
		for _, id := range spec.TagIDs {
			o := tags.AttachedObjects{TagID: id}
			for i := range s.Association[id] {
				o.ObjectIDs = append(o.ObjectIDs, i)
			}
			res = append(res, o)
		}
		OK(w, res)

	case "list-attached-tags-on-objects":
		var res []tags.AttachedTags
		for _, ref := range spec.ObjectIDs {
			o := tags.AttachedTags{ObjectID: ref}
			for id, objs := range s.Association {
				if objs[ref] {
					o.TagIDs = append(o.TagIDs, id)
				}
			}
			res = append(res, o)
		}
		OK(w, res)

	case "attach-multiple-tags-to-object":
		// TODO: add check if target (moref) exist or return 403 as per API behavior

		res := struct {
			Success bool             `json:"success"`
			Errors  tags.BatchErrors `json:"error_messages,omitempty"`
		}{}

		for _, id := range spec.TagIDs {
			if _, exists := s.Association[id]; !exists {
				log.Printf("association tag not found: %s", id)
				res.Errors = append(res.Errors, tags.BatchError{
					Type:    "cis.tagging.objectNotFound.error",
					Message: fmt.Sprintf("Tagging object %s not found", id),
				})
			} else {
				s.Association[id][*spec.ObjectID] = true
			}
		}

		if len(res.Errors) == 0 {
			res.Success = true
		}
		OK(w, res)

	case "detach-multiple-tags-from-object":
		// TODO: add check if target (moref) exist or return 403 as per API behavior

		res := struct {
			Success bool             `json:"success"`
			Errors  tags.BatchErrors `json:"error_messages,omitempty"`
		}{}

		for _, id := range spec.TagIDs {
			if _, exists := s.Association[id]; !exists {
				log.Printf("association tag not found: %s", id)
				res.Errors = append(res.Errors, tags.BatchError{
					Type:    "cis.tagging.objectNotFound.error",
					Message: fmt.Sprintf("Tagging object %s not found", id),
				})
			} else {
				s.Association[id][*spec.ObjectID] = false
			}
		}

		if len(res.Errors) == 0 {
			res.Success = true
		}
		OK(w, res)
	}
}

func (s *handler) associationID(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	id := s.id(r)
	if _, exists := s.Association[id]; !exists {
		log.Printf("association tag not found: %s", id)
		http.NotFound(w, r)
		return
	}

	var spec internal.Association
	var specs struct {
		ObjectIDs []internal.AssociatedObject `json:"object_ids"`
	}
	switch s.action(r) {
	case "attach", "detach", "list-attached-objects":
		if !s.decode(r, w, &spec) {
			return
		}
	case "attach-tag-to-multiple-objects":
		if !s.decode(r, w, &specs) {
			return
		}
	}

	switch s.action(r) {
	case "attach":
		s.Association[id][*spec.ObjectID] = true
		OK(w)
	case "detach":
		delete(s.Association[id], *spec.ObjectID)
		OK(w)
	case "list-attached-objects":
		var ids []internal.AssociatedObject
		for id := range s.Association[id] {
			ids = append(ids, id)
		}
		OK(w, ids)
	case "attach-tag-to-multiple-objects":
		for _, obj := range specs.ObjectIDs {
			s.Association[id][obj] = true
		}
		OK(w)
	}
}

func (s *handler) library(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodPost:
		var spec struct {
			Library library.Library `json:"create_spec"`
			Find    library.Find    `json:"spec"`
		}
		if !s.decode(r, w, &spec) {
			return
		}

		switch s.action(r) {
		case "find":
			var ids []string
			for _, l := range s.Library {
				if spec.Find.Type != "" {
					if spec.Find.Type != l.Library.Type {
						continue
					}
				}
				if spec.Find.Name != "" {
					if !strings.EqualFold(l.Library.Name, spec.Find.Name) {
						continue
					}
				}
				ids = append(ids, l.ID)
			}
			OK(w, ids)
		case "":
			if !s.isValidSecurityPolicy(spec.Library.SecurityPolicyID) {
				http.NotFound(w, r)
				return
			}

			id := uuid.New().String()
			spec.Library.ID = id
			spec.Library.CreationTime = types.NewTime(time.Now())
			spec.Library.LastModifiedTime = types.NewTime(time.Now())
			spec.Library.UnsetSecurityPolicyID = spec.Library.SecurityPolicyID == ""
			dir := libraryPath(&spec.Library, "")
			if err := os.Mkdir(dir, 0750); err != nil {
				s.error(w, err)
				return
			}
			s.Library[id] = &content{
				Library: &spec.Library,
				Item:    make(map[string]*item),
				Subs:    make(map[string]*library.Subscriber),
				VMTX:    make(map[string]*types.ManagedObjectReference),
			}

			pub := spec.Library.Publication
			if pub != nil && pub.Published != nil && *pub.Published {
				// Generate PublishURL as real vCenter does
				pub.PublishURL = (&url.URL{
					Scheme: s.URL.Scheme,
					Host:   s.URL.Host,
					Path:   "/cls/vcsp/lib/" + id,
				}).String()
			}

			sub := spec.Library.Subscription
			if sub != nil {
				// Share the published Item map
				pid := path.Base(sub.SubscriptionURL)
				if p, ok := s.Library[pid]; ok {
					s.Library[id].Item = p.Item
				}
			}

			OK(w, id)
		}
	case http.MethodGet:
		var ids []string
		for id := range s.Library {
			ids = append(ids, id)
		}
		OK(w, ids)
	}
}

func (s *handler) publish(w http.ResponseWriter, r *http.Request, sids []internal.SubscriptionDestination, l *content, vmtx *item) bool {
	var ids []string
	if len(sids) == 0 {
		for sid := range l.Subs {
			ids = append(ids, sid)
		}
	} else {
		for _, dst := range sids {
			ids = append(ids, dst.ID)
		}
	}

	for _, sid := range ids {
		sub, ok := l.Subs[sid]
		if !ok {
			log.Printf("library subscription not found: %s", sid)
			http.NotFound(w, r)
			return false
		}

		slib := s.Library[sub.LibraryID]
		if slib.VMTX[vmtx.ID] != nil {
			return true // already cloned
		}

		ds := &vcenter.DiskStorage{Datastore: l.Library.Storage[0].DatastoreID}
		ref, err := s.cloneVM(vmtx.Template.Value, vmtx.Name, sub.Placement, ds)
		if err != nil {
			s.error(w, err)
			return false
		}

		slib.VMTX[vmtx.ID] = ref
	}

	return true
}

func (s *handler) libraryID(w http.ResponseWriter, r *http.Request) {
	id := s.id(r)
	l, ok := s.Library[id]
	if !ok {
		log.Printf("library not found: %s", id)
		http.NotFound(w, r)
		return
	}

	switch r.Method {
	case http.MethodDelete:
		p := libraryPath(l.Library, "")
		if err := os.RemoveAll(p); err != nil {
			s.error(w, err)
			return
		}
		for _, item := range l.Item {
			s.deleteVM(item.Template)
		}
		delete(s.Library, id)
		OK(w)
	case http.MethodPatch:
		var spec struct {
			Library library.Library `json:"update_spec"`
		}
		if s.decode(r, w, &spec) {
			l.Patch(&spec.Library)
			OK(w)
		}
	case http.MethodPost:
		switch s.action(r) {
		case "publish":
			var spec internal.SubscriptionDestinationSpec
			if !s.decode(r, w, &spec) {
				return
			}
			for _, item := range l.Item {
				if item.Type != library.ItemTypeVMTX {
					continue
				}
				if !s.publish(w, r, spec.Subscriptions, l, item) {
					return
				}
			}
			OK(w)
		case "sync":
			if l.Type == "SUBSCRIBED" {
				l.LastSyncTime = types.NewTime(time.Now())
				OK(w)
			} else {
				http.NotFound(w, r)
			}
		}
	case http.MethodGet:
		OK(w, l)
	}
}

func (s *handler) subscriptions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	id := r.URL.Query().Get("library")
	l, ok := s.Library[id]
	if !ok {
		log.Printf("library not found: %s", id)
		http.NotFound(w, r)
		return
	}

	var res []library.SubscriberSummary
	for sid, slib := range l.Subs {
		res = append(res, library.SubscriberSummary{
			LibraryID:              slib.LibraryID,
			LibraryName:            slib.LibraryName,
			SubscriptionID:         sid,
			LibraryVcenterHostname: "",
		})
	}
	OK(w, res)
}

func (s *handler) subscriptionsID(w http.ResponseWriter, r *http.Request) {
	id := s.id(r)
	l, ok := s.Library[id]
	if !ok {
		log.Printf("library not found: %s", id)
		http.NotFound(w, r)
		return
	}

	switch s.action(r) {
	case "get":
		var dst internal.SubscriptionDestination
		if !s.decode(r, w, &dst) {
			return
		}

		sub, ok := l.Subs[dst.ID]
		if !ok {
			log.Printf("library subscription not found: %s", dst.ID)
			http.NotFound(w, r)
			return
		}

		OK(w, sub)
	case "delete":
		var dst internal.SubscriptionDestination
		if !s.decode(r, w, &dst) {
			return
		}

		delete(l.Subs, dst.ID)

		OK(w)
	case "create", "":
		var spec struct {
			Sub struct {
				SubscriberLibrary library.SubscriberLibrary `json:"subscribed_library"`
			} `json:"spec"`
		}
		if !s.decode(r, w, &spec) {
			return
		}

		sub := spec.Sub.SubscriberLibrary
		slib, ok := s.Library[sub.LibraryID]
		if !ok {
			log.Printf("library not found: %s", sub.LibraryID)
			http.NotFound(w, r)
			return
		}

		id := uuid.New().String()
		l.Subs[id] = &library.Subscriber{
			LibraryID:       slib.ID,
			LibraryName:     slib.Name,
			LibraryLocation: sub.Target,
			Placement:       sub.Placement,
			Vcenter:         sub.Vcenter,
		}

		OK(w, id)
	}
}

func (s *handler) libraryItem(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodPost:
		var spec struct {
			Item library.Item     `json:"create_spec"`
			Find library.FindItem `json:"spec"`
		}
		if !s.decode(r, w, &spec) {
			return
		}

		switch s.action(r) {
		case "find":
			var ids []string
			for _, l := range s.Library {
				if spec.Find.LibraryID != "" {
					if spec.Find.LibraryID != l.ID {
						continue
					}
				}
				for _, i := range l.Item {
					if spec.Find.Name != "" {
						if spec.Find.Name != i.Name {
							continue
						}
					}
					if spec.Find.Type != "" {
						if spec.Find.Type != i.Type {
							continue
						}
					}
					ids = append(ids, i.ID)
				}
			}
			OK(w, ids)
		case "create", "":
			id := spec.Item.LibraryID
			l, ok := s.Library[id]
			if !ok {
				log.Printf("library not found: %s", id)
				http.NotFound(w, r)
				return
			}
			if l.Type == "SUBSCRIBED" {
				BadRequest(w, "com.vmware.vapi.std.errors.invalid_element_type")
				return
			}
			for _, item := range l.Item {
				if item.Name == spec.Item.Name {
					BadRequest(w, "com.vmware.vapi.std.errors.already_exists")
					return
				}
			}
			id = uuid.New().String()
			spec.Item.ID = id
			spec.Item.CreationTime = types.NewTime(time.Now())
			spec.Item.LastModifiedTime = types.NewTime(time.Now())
			if l.SecurityPolicyID != "" {
				// TODO: verify signed items
				spec.Item.SecurityCompliance = types.NewBool(false)
				spec.Item.CertificateVerification = &library.ItemCertificateVerification{
					Status: "NOT_AVAILABLE",
				}
			}
			l.Item[id] = &item{Item: &spec.Item}
			OK(w, id)
		}
	case http.MethodGet:
		id := r.URL.Query().Get("library_id")
		l, ok := s.Library[id]
		if !ok {
			log.Printf("library not found: %s", id)
			http.NotFound(w, r)
			return
		}

		var ids []string
		for id := range l.Item {
			ids = append(ids, id)
		}
		OK(w, ids)
	}
}

func (s *handler) libraryItemID(w http.ResponseWriter, r *http.Request) {
	id := s.id(r)
	lid := r.URL.Query().Get("library_id")
	if lid == "" {
		if l := s.itemLibrary(id); l != nil {
			lid = l.ID
		}
	}
	l, ok := s.Library[lid]
	if !ok {
		log.Printf("library not found: %q", lid)
		http.NotFound(w, r)
		return
	}
	item, ok := l.Item[id]
	if !ok {
		log.Printf("library item not found: %q", id)
		http.NotFound(w, r)
		return
	}

	switch r.Method {
	case http.MethodDelete:
		p := libraryPath(l.Library, id)
		if err := os.RemoveAll(p); err != nil {
			s.error(w, err)
			return
		}
		s.deleteVM(l.Item[item.ID].Template)
		delete(l.Item, item.ID)
		OK(w)
	case http.MethodPatch:
		var spec struct {
			library.Item `json:"update_spec"`
		}
		if s.decode(r, w, &spec) {
			item.Patch(&spec.Item)
			OK(w)
		}
	case http.MethodPost:
		switch s.action(r) {
		case "copy":
			var spec struct {
				library.Item `json:"destination_create_spec"`
			}
			if !s.decode(r, w, &spec) {
				return
			}

			l, ok = s.Library[spec.LibraryID]
			if !ok {
				log.Printf("library not found: %q", spec.LibraryID)
				http.NotFound(w, r)
				return
			}
			if spec.Name == "" {
				BadRequest(w, "com.vmware.vapi.std.errors.invalid_argument")
			}

			id := uuid.New().String()
			nitem := item.cp()
			nitem.ID = id
			nitem.LibraryID = spec.LibraryID
			l.Item[id] = nitem

			OK(w, id)
		case "sync":
			if l.Type == "SUBSCRIBED" {
				item.LastSyncTime = types.NewTime(time.Now())
				OK(w)
			} else {
				http.NotFound(w, r)
			}
		case "publish":
			var spec internal.SubscriptionDestinationSpec
			if s.decode(r, w, &spec) {
				if s.publish(w, r, spec.Subscriptions, l, item) {
					OK(w)
				}
			}
		}
	case http.MethodGet:
		OK(w, item)
	}
}

func (s *handler) libraryItemUpdateSession(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		var ids []string
		for id := range s.Update {
			ids = append(ids, id)
		}
		OK(w, ids)
	case http.MethodPost:
		var spec struct {
			Session library.Session `json:"create_spec"`
		}
		if !s.decode(r, w, &spec) {
			return
		}

		switch s.action(r) {
		case "create", "":
			lib := s.itemLibrary(spec.Session.LibraryItemID)
			if lib == nil {
				log.Printf("library for item %q not found", spec.Session.LibraryItemID)
				http.NotFound(w, r)
				return
			}
			session := &library.Session{
				ID:                        uuid.New().String(),
				LibraryItemID:             spec.Session.LibraryItemID,
				LibraryItemContentVersion: "1",
				ClientProgress:            0,
				State:                     "ACTIVE",
				ExpirationTime:            types.NewTime(time.Now().Add(time.Hour)),
			}
			s.Update[session.ID] = update{
				Session: session,
				Library: lib,
				File:    make(map[string]*library.UpdateFile),
			}
			OK(w, session.ID)
		}
	}
}

func (s *handler) libraryItemUpdateSessionID(w http.ResponseWriter, r *http.Request) {
	id := s.id(r)
	up, ok := s.Update[id]
	if !ok {
		log.Printf("update session not found: %s", id)
		http.NotFound(w, r)
		return
	}

	session := up.Session
	done := func(state string) {
		up.State = state
		go time.AfterFunc(session.ExpirationTime.Sub(time.Now()), func() {
			s.Lock()
			delete(s.Update, id)
			s.Unlock()
		})
	}

	switch r.Method {
	case http.MethodGet:
		OK(w, session)
	case http.MethodPost:
		switch s.action(r) {
		case "cancel":
			done("CANCELED")
		case "complete":
			done("DONE")
		case "fail":
			done("ERROR")
		case "keep-alive":
			session.ExpirationTime = types.NewTime(time.Now().Add(time.Hour))
		}
		OK(w)
	case http.MethodDelete:
		delete(s.Update, id)
		OK(w)
	}
}

func (s *handler) libraryItemUpdateSessionFile(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	id := r.URL.Query().Get("update_session_id")
	up, ok := s.Update[id]
	if !ok {
		log.Printf("update session not found: %s", id)
		http.NotFound(w, r)
		return
	}

	var files []*library.UpdateFile
	for _, f := range up.File {
		files = append(files, f)
	}
	OK(w, files)
}

func (s *handler) pullSource(up update, info *library.UpdateFile) {
	done := func(err error) {
		s.Lock()
		info.Status = "READY"
		if err != nil {
			log.Printf("PULL %s: %s", info.SourceEndpoint.URI, err)
			info.Status = "ERROR"
			up.State = "ERROR"
			up.ErrorMessage = &rest.LocalizableMessage{DefaultMessage: err.Error()}
		}
		s.Unlock()
	}

	c := &http.Client{
		Transport: &http.Transport{
			TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
		},
	}

	res, err := c.Get(info.SourceEndpoint.URI)
	if err != nil {
		done(err)
		return
	}

	err = s.libraryItemFileCreate(&up, info.Name, res.Body)
	done(err)
}

func (s *handler) libraryItemUpdateSessionFileID(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	id := s.id(r)
	up, ok := s.Update[id]
	if !ok {
		log.Printf("update session not found: %s", id)
		http.NotFound(w, r)
		return
	}

	switch s.action(r) {
	case "add":
		var spec struct {
			File library.UpdateFile `json:"file_spec"`
		}
		if s.decode(r, w, &spec) {
			id = uuid.New().String()
			info := &library.UpdateFile{
				Name:             spec.File.Name,
				SourceType:       spec.File.SourceType,
				Status:           "WAITING_FOR_TRANSFER",
				BytesTransferred: 0,
			}
			switch info.SourceType {
			case "PUSH":
				u := url.URL{
					Scheme: s.URL.Scheme,
					Host:   s.URL.Host,
					Path:   path.Join(rest.Path, internal.LibraryItemFileData, id, info.Name),
				}
				info.UploadEndpoint = &library.TransferEndpoint{URI: u.String()}
			case "PULL":
				info.SourceEndpoint = spec.File.SourceEndpoint
				go s.pullSource(up, info)
			}
			up.File[id] = info
			OK(w, info)
		}
	case "get":
		OK(w, up.Session)
	case "list":
		var ids []string
		for id := range up.File {
			ids = append(ids, id)
		}
		OK(w, ids)
	case "remove":
		delete(s.Update, id)
		OK(w)
	case "validate":
		// TODO
	}
}

func (s *handler) libraryItemDownloadSession(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		var ids []string
		for id := range s.Download {
			ids = append(ids, id)
		}
		OK(w, ids)
	case http.MethodPost:
		var spec struct {
			Session library.Session `json:"create_spec"`
		}
		if !s.decode(r, w, &spec) {
			return
		}

		switch s.action(r) {
		case "create", "":
			var lib *library.Library
			var files []library.File
			for _, l := range s.Library {
				if item, ok := l.Item[spec.Session.LibraryItemID]; ok {
					lib = l.Library
					files = item.File
					break
				}
			}
			if lib == nil {
				log.Printf("library for item %q not found", spec.Session.LibraryItemID)
				http.NotFound(w, r)
				return
			}
			session := &library.Session{
				ID:                        uuid.New().String(),
				LibraryItemID:             spec.Session.LibraryItemID,
				LibraryItemContentVersion: "1",
				ClientProgress:            0,
				State:                     "ACTIVE",
				ExpirationTime:            types.NewTime(time.Now().Add(time.Hour)),
			}
			s.Download[session.ID] = download{
				Session: session,
				Library: lib,
				File:    make(map[string]*library.DownloadFile),
			}
			for _, file := range files {
				s.Download[session.ID].File[file.Name] = &library.DownloadFile{
					Name:   file.Name,
					Status: "UNPREPARED",
				}
			}
			OK(w, session.ID)
		}
	}
}

func (s *handler) libraryItemDownloadSessionID(w http.ResponseWriter, r *http.Request) {
	id := s.id(r)
	up, ok := s.Download[id]
	if !ok {
		log.Printf("download session not found: %s", id)
		http.NotFound(w, r)
		return
	}

	session := up.Session
	switch r.Method {
	case http.MethodGet:
		OK(w, session)
	case http.MethodPost:
		switch s.action(r) {
		case "cancel", "complete", "fail":
			delete(s.Download, id) // TODO: fully mock VC's behavior
		case "keep-alive":
			session.ExpirationTime = types.NewTime(time.Now().Add(time.Hour))
		}
		OK(w)
	case http.MethodDelete:
		delete(s.Download, id)
		OK(w)
	}
}

func (s *handler) libraryItemDownloadSessionFile(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	id := r.URL.Query().Get("download_session_id")
	dl, ok := s.Download[id]
	if !ok {
		log.Printf("download session not found: %s", id)
		http.NotFound(w, r)
		return
	}

	var files []*library.DownloadFile
	for _, f := range dl.File {
		files = append(files, f)
	}
	OK(w, files)
}

func (s *handler) libraryItemDownloadSessionFileID(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	id := s.id(r)
	dl, ok := s.Download[id]
	if !ok {
		log.Printf("download session not found: %s", id)
		http.NotFound(w, r)
		return
	}

	var spec struct {
		File string `json:"file_name"`
	}

	switch s.action(r) {
	case "prepare":
		if s.decode(r, w, &spec) {
			u := url.URL{
				Scheme: s.URL.Scheme,
				Host:   s.URL.Host,
				Path:   path.Join(rest.Path, internal.LibraryItemFileData, id, spec.File),
			}
			info := &library.DownloadFile{
				Name:             spec.File,
				Status:           "PREPARED",
				BytesTransferred: 0,
				DownloadEndpoint: &library.TransferEndpoint{
					URI: u.String(),
				},
			}
			dl.File[spec.File] = info
			OK(w, info)
		}
	case "get":
		if s.decode(r, w, &spec) {
			OK(w, dl.File[spec.File])
		}
	}
}

func (s *handler) itemLibrary(id string) *library.Library {
	for _, l := range s.Library {
		if _, ok := l.Item[id]; ok {
			return l.Library
		}
	}
	return nil
}

func (s *handler) updateFileInfo(id string) *update {
	for _, up := range s.Update {
		for i := range up.File {
			if i == id {
				return &up
			}
		}
	}
	return nil
}

// libraryPath returns the local Datastore fs path for a Library or Item if id is specified.
func libraryPath(l *library.Library, id string) string {
	dsref := types.ManagedObjectReference{
		Type:  "Datastore",
		Value: l.Storage[0].DatastoreID,
	}
	ds := simulator.Map.Get(dsref).(*simulator.Datastore)

	return path.Join(append([]string{ds.Info.GetDatastoreInfo().Url, "contentlib-" + l.ID}, id)...)
}

func (s *handler) libraryItemFileCreate(up *update, name string, body io.ReadCloser) error {
	var in io.Reader = body
	dir := libraryPath(up.Library, up.Session.LibraryItemID)
	if err := os.MkdirAll(dir, 0750); err != nil {
		return err
	}

	if path.Ext(name) == ".ova" {
		// All we need is the .ovf, vcsim has no use for .vmdk or .mf
		r := tar.NewReader(body)
		for {
			h, err := r.Next()
			if err != nil {
				return err
			}

			if path.Ext(h.Name) == ".ovf" {
				name = h.Name
				in = io.LimitReader(body, h.Size)
				break
			}
		}
	}

	file, err := os.Create(path.Join(dir, name))
	if err != nil {
		return err
	}

	n, err := io.Copy(file, in)
	_ = body.Close()
	if err != nil {
		return err
	}
	err = file.Close()
	if err != nil {
		return err
	}

	i := s.Library[up.Library.ID].Item[up.Session.LibraryItemID]
	i.File = append(i.File, library.File{
		Cached:  types.NewBool(true),
		Name:    name,
		Size:    types.NewInt64(n),
		Version: "1",
	})

	return nil
}

func (s *handler) libraryItemFileData(w http.ResponseWriter, r *http.Request) {
	p := strings.Split(r.URL.Path, "/")
	id, name := p[len(p)-2], p[len(p)-1]

	if r.Method == http.MethodGet {
		dl, ok := s.Download[id]
		if !ok {
			log.Printf("library download not found: %s", id)
			http.NotFound(w, r)
			return
		}
		p := path.Join(libraryPath(dl.Library, dl.Session.LibraryItemID), name)
		f, err := os.Open(p)
		if err != nil {
			s.error(w, err)
			return
		}
		_, err = io.Copy(w, f)
		if err != nil {
			log.Printf("copy %s: %s", p, err)
		}
		_ = f.Close()
		return
	}

	if r.Method != http.MethodPut {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	up := s.updateFileInfo(id)
	if up == nil {
		log.Printf("library update not found: %s", id)
		http.NotFound(w, r)
		return
	}

	err := s.libraryItemFileCreate(up, name, r.Body)
	if err != nil {
		s.error(w, err)
	}
}

func (s *handler) libraryItemFile(w http.ResponseWriter, r *http.Request) {
	id := r.URL.Query().Get("library_item_id")
	for _, l := range s.Library {
		if i, ok := l.Item[id]; ok {
			OK(w, i.File)
			return
		}
	}
	http.NotFound(w, r)
}

func (s *handler) libraryItemFileID(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	id := s.id(r)
	var spec struct {
		Name string `json:"name"`
	}
	if !s.decode(r, w, &spec) {
		return
	}
	for _, l := range s.Library {
		if i, ok := l.Item[id]; ok {
			for _, f := range i.File {
				if f.Name == spec.Name {
					OK(w, f)
					return
				}
			}
		}
	}
	http.NotFound(w, r)
}

func (i *item) cp() *item {
	nitem := *i.Item
	return &item{&nitem, i.File, i.Template}
}

func (i *item) ovf() string {
	for _, f := range i.File {
		if strings.HasSuffix(f.Name, ".ovf") {
			return f.Name
		}
	}
	return ""
}

func vmConfigSpec(ctx context.Context, c *vim25.Client, deploy vcenter.Deploy) (*types.VirtualMachineConfigSpec, error) {
	if deploy.VmConfigSpec == nil {
		return nil, nil
	}

	b, err := base64.StdEncoding.DecodeString(deploy.VmConfigSpec.XML)
	if err != nil {
		return nil, err
	}

	var spec *types.VirtualMachineConfigSpec

	dec := xml.NewDecoder(bytes.NewReader(b))
	dec.TypeFunc = c.Types
	err = dec.Decode(&spec)
	if err != nil {
		return nil, err
	}

	return spec, nil
}

func (s *handler) libraryDeploy(ctx context.Context, c *vim25.Client, lib *library.Library, item *item, deploy vcenter.Deploy) (*nfc.LeaseInfo, error) {
	config, err := vmConfigSpec(ctx, c, deploy)
	if err != nil {
		return nil, err
	}

	name := item.ovf()
	desc, err := ioutil.ReadFile(filepath.Join(libraryPath(lib, item.ID), name))
	if err != nil {
		return nil, err
	}
	ds := types.ManagedObjectReference{Type: "Datastore", Value: deploy.DeploymentSpec.DefaultDatastoreID}
	pool := types.ManagedObjectReference{Type: "ResourcePool", Value: deploy.Target.ResourcePoolID}
	var folder, host *types.ManagedObjectReference
	if deploy.Target.FolderID != "" {
		folder = &types.ManagedObjectReference{Type: "Folder", Value: deploy.Target.FolderID}
	}
	if deploy.Target.HostID != "" {
		host = &types.ManagedObjectReference{Type: "HostSystem", Value: deploy.Target.HostID}
	}

	v, err := view.NewManager(c).CreateContainerView(ctx, c.ServiceContent.RootFolder, nil, true)
	if err != nil {
		return nil, err
	}
	defer func() {
		_ = v.Destroy(ctx)
	}()
	refs, err := v.Find(ctx, []string{"Network"}, nil)
	if err != nil {
		return nil, err
	}

	var network []types.OvfNetworkMapping
	for _, net := range deploy.NetworkMappings {
		for i := range refs {
			if refs[i].Value == net.Value {
				network = append(network, types.OvfNetworkMapping{Name: net.Key, Network: refs[i]})
				break
			}
		}
	}

	if ds.Value == "" {
		// Datastore is optional in the deploy spec, but not in OvfManager.CreateImportSpec
		refs, err = v.Find(ctx, []string{"Datastore"}, nil)
		if err != nil {
			return nil, err
		}
		// TODO: consider StorageProfileID
		ds = refs[0]
	}

	cisp := types.OvfCreateImportSpecParams{
		DiskProvisioning: deploy.DeploymentSpec.StorageProvisioning,
		EntityName:       deploy.DeploymentSpec.Name,
		NetworkMapping:   network,
	}

	for _, p := range deploy.AdditionalParams {
		switch p.Type {
		case vcenter.TypePropertyParams:
			for _, prop := range p.Properties {
				cisp.PropertyMapping = append(cisp.PropertyMapping, types.KeyValue{
					Key:   prop.ID,
					Value: prop.Value,
				})
			}
		case vcenter.TypeDeploymentOptionParams:
			cisp.OvfManagerCommonParams.DeploymentOption = p.SelectedKey
		}
	}

	m := ovf.NewManager(c)
	spec, err := m.CreateImportSpec(ctx, string(desc), pool, ds, cisp)
	if err != nil {
		return nil, err
	}
	if spec.Error != nil {
		return nil, errors.New(spec.Error[0].LocalizedMessage)
	}

	req := types.ImportVApp{
		This:   pool,
		Spec:   spec.ImportSpec,
		Folder: folder,
		Host:   host,
	}
	res, err := methods.ImportVApp(ctx, c, &req)
	if err != nil {
		return nil, err
	}

	lease := nfc.NewLease(c, res.Returnval)
	info, err := lease.Wait(ctx, spec.FileItem)
	if err != nil {
		return nil, err
	}

	if err = lease.Complete(ctx); err != nil {
		return nil, err
	}

	if config != nil {
		if err = s.reconfigVM(info.Entity, *config); err != nil {
			return nil, err
		}
	}

	return info, nil
}

func (s *handler) libraryItemOVF(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	var req vcenter.OVF
	if !s.decode(r, w, &req) {
		return
	}

	switch {
	case req.Target.LibraryItemID != "":
	case req.Target.LibraryID != "":
		l, ok := s.Library[req.Target.LibraryID]
		if !ok {
			http.NotFound(w, r)
		}

		id := uuid.New().String()
		l.Item[id] = &item{
			Item: &library.Item{
				ID:               id,
				LibraryID:        l.Library.ID,
				Name:             req.Spec.Name,
				Description:      req.Spec.Description,
				Type:             library.ItemTypeOVF,
				CreationTime:     types.NewTime(time.Now()),
				LastModifiedTime: types.NewTime(time.Now()),
			},
		}

		res := vcenter.CreateResult{
			Succeeded: true,
			ID:        id,
		}
		OK(w, res)
	default:
		BadRequest(w, "com.vmware.vapi.std.errors.invalid_argument")
		return
	}
}

func (s *handler) libraryItemOVFID(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	id := s.id(r)
	ok := false
	var lib *library.Library
	var item *item
	for _, l := range s.Library {
		if l.Library.Type == "SUBSCRIBED" {
			// Subscribers share the same Item map, we need the LOCAL library to find the .ovf on disk
			continue
		}
		item, ok = l.Item[id]
		if ok {
			lib = l.Library
			break
		}
	}
	if !ok {
		log.Printf("library item not found: %q", id)
		http.NotFound(w, r)
		return
	}

	var spec struct {
		vcenter.Deploy
	}
	if !s.decode(r, w, &spec) {
		return
	}

	switch s.action(r) {
	case "deploy":
		var d vcenter.Deployment
		err := s.withClient(func(ctx context.Context, c *vim25.Client) error {
			info, err := s.libraryDeploy(ctx, c, lib, item, spec.Deploy)
			if err != nil {
				return err
			}
			id := vcenter.ResourceID(info.Entity)
			d.Succeeded = true
			d.ResourceID = &id
			return nil
		})
		if err != nil {
			d.Error = &vcenter.DeploymentError{
				Errors: []vcenter.OVFError{{
					Category: "SERVER",
					Error: &vcenter.Error{
						Class: "com.vmware.vapi.std.errors.error",
						Messages: []rest.LocalizableMessage{
							{
								DefaultMessage: err.Error(),
							},
						},
					},
				}},
			}
		}
		OK(w, d)
	case "filter":
		res := vcenter.FilterResponse{
			Name: item.Name,
		}
		OK(w, res)
	default:
		http.NotFound(w, r)
	}
}

func (s *handler) deleteVM(ref *types.ManagedObjectReference) {
	if ref == nil {
		return
	}
	_ = s.withClient(func(ctx context.Context, c *vim25.Client) error {
		_, _ = object.NewVirtualMachine(c, *ref).Destroy(ctx)
		return nil
	})
}

func (s *handler) reconfigVM(ref types.ManagedObjectReference, config types.VirtualMachineConfigSpec) error {
	return s.withClient(func(ctx context.Context, c *vim25.Client) error {
		vm := object.NewVirtualMachine(c, ref)
		task, err := vm.Reconfigure(ctx, config)
		if err != nil {
			return err
		}
		return task.Wait(ctx)
	})
}

func (s *handler) cloneVM(source string, name string, p *library.Placement, storage *vcenter.DiskStorage) (*types.ManagedObjectReference, error) {
	var folder, pool, host, ds *types.ManagedObjectReference
	if p.Folder != "" {
		folder = &types.ManagedObjectReference{Type: "Folder", Value: p.Folder}
	}
	if p.ResourcePool != "" {
		pool = &types.ManagedObjectReference{Type: "ResourcePool", Value: p.ResourcePool}
	}
	if p.Host != "" {
		host = &types.ManagedObjectReference{Type: "HostSystem", Value: p.Host}
	}
	if storage != nil {
		if storage.Datastore != "" {
			ds = &types.ManagedObjectReference{Type: "Datastore", Value: storage.Datastore}
		}
	}

	spec := types.VirtualMachineCloneSpec{
		Template: true,
		Location: types.VirtualMachineRelocateSpec{
			Folder:    folder,
			Pool:      pool,
			Host:      host,
			Datastore: ds,
		},
	}

	var ref *types.ManagedObjectReference

	return ref, s.withClient(func(ctx context.Context, c *vim25.Client) error {
		vm := object.NewVirtualMachine(c, types.ManagedObjectReference{Type: "VirtualMachine", Value: source})

		task, err := vm.Clone(ctx, object.NewFolder(c, *folder), name, spec)
		if err != nil {
			return err
		}
		res, err := task.WaitForResult(ctx, nil)
		if err != nil {
			return err
		}
		ref = types.NewReference(res.Result.(types.ManagedObjectReference))
		return nil
	})
}

func (s *handler) libraryItemCreateTemplate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	var spec struct {
		vcenter.Template `json:"spec"`
	}
	if !s.decode(r, w, &spec) {
		return
	}

	l, ok := s.Library[spec.Library]
	if !ok {
		http.NotFound(w, r)
		return
	}

	ds := &vcenter.DiskStorage{Datastore: l.Library.Storage[0].DatastoreID}
	ref, err := s.cloneVM(spec.SourceVM, spec.Name, spec.Placement, ds)
	if err != nil {
		BadRequest(w, err.Error())
		return
	}

	id := uuid.New().String()
	l.Item[id] = &item{
		Item: &library.Item{
			ID:               id,
			LibraryID:        l.Library.ID,
			Name:             spec.Name,
			Type:             library.ItemTypeVMTX,
			CreationTime:     types.NewTime(time.Now()),
			LastModifiedTime: types.NewTime(time.Now()),
		},
		Template: ref,
	}

	OK(w, id)
}

func (s *handler) libraryItemTemplateID(w http.ResponseWriter, r *http.Request) {
	// Go's ServeMux doesn't support wildcard matching, hacking around that for now to support
	// CheckOuts, e.g. "/vcenter/vm-template/library-items/{item}/check-outs/{vm}?action=check-in"
	p := strings.TrimPrefix(r.URL.Path, rest.Path+internal.VCenterVMTXLibraryItem+"/")
	route := strings.Split(p, "/")
	if len(route) == 0 {
		http.NotFound(w, r)
		return
	}

	id := route[0]
	ok := false

	var item *item
	for _, l := range s.Library {
		item, ok = l.Item[id]
		if ok {
			break
		}
	}
	if !ok {
		log.Printf("library item not found: %q", id)
		http.NotFound(w, r)
		return
	}

	if item.Type != library.ItemTypeVMTX {
		BadRequest(w, "com.vmware.vapi.std.errors.invalid_argument")
		return
	}

	if len(route) > 1 {
		switch route[1] {
		case "check-outs":
			s.libraryItemCheckOuts(item, w, r)
			return
		default:
			http.NotFound(w, r)
			return
		}
	}

	if r.Method == http.MethodGet {
		// TODO: add mock data
		t := &vcenter.TemplateInfo{}
		OK(w, t)
		return
	}

	var spec struct {
		vcenter.DeployTemplate `json:"spec"`
	}
	if !s.decode(r, w, &spec) {
		return
	}

	switch r.URL.Query().Get("action") {
	case "deploy":
		p := spec.Placement
		if p == nil {
			BadRequest(w, "com.vmware.vapi.std.errors.invalid_argument")
			return
		}
		if p.Cluster == "" && p.Host == "" && p.ResourcePool == "" {
			BadRequest(w, "com.vmware.vapi.std.errors.invalid_argument")
			return
		}

		ref, err := s.cloneVM(item.Template.Value, spec.Name, p, spec.DiskStorage)
		if err != nil {
			BadRequest(w, err.Error())
			return
		}
		OK(w, ref.Value)
	default:
		http.NotFound(w, r)
	}
}

func (s *handler) libraryItemCheckOuts(item *item, w http.ResponseWriter, r *http.Request) {
	switch r.URL.Query().Get("action") {
	case "check-out":
		var spec struct {
			*vcenter.CheckOut `json:"spec"`
		}
		if !s.decode(r, w, &spec) {
			return
		}

		ref, err := s.cloneVM(item.Template.Value, spec.Name, spec.Placement, nil)
		if err != nil {
			BadRequest(w, err.Error())
			return
		}
		OK(w, ref.Value)
	case "check-in":
		// TODO: increment ContentVersion
		OK(w, "0")
	default:
		http.NotFound(w, r)
	}
}

// defaultSecurityPolicies generates the initial set of security policies always present on vCenter.
func defaultSecurityPolicies() []library.ContentSecurityPoliciesInfo {
	policyID, _ := uuid.NewUUID()
	return []library.ContentSecurityPoliciesInfo{
		{
			ItemTypeRules: map[string]string{
				"ovf": "OVF_STRICT_VERIFICATION",
			},
			Name:   "OVF default policy",
			Policy: policyID.String(),
		},
	}
}

func (s *handler) librarySecurityPolicies(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		StatusOK(w, s.Policies)
	default:
		w.WriteHeader(http.StatusMethodNotAllowed)
	}
}

func (s *handler) isValidSecurityPolicy(policy string) bool {
	if policy == "" {
		return true
	}

	for _, p := range s.Policies {
		if p.Policy == policy {
			return true
		}
	}
	return false
}

func (s *handler) libraryTrustedCertificates(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		var res struct {
			Certificates []library.TrustedCertificateSummary `json:"certificates"`
		}
		for id, cert := range s.Trust {
			res.Certificates = append(res.Certificates, library.TrustedCertificateSummary{
				TrustedCertificate: cert,
				ID:                 id,
			})
		}

		StatusOK(w, &res)
	case http.MethodPost:
		var info library.TrustedCertificate
		if s.decode(r, w, &info) {
			block, _ := pem.Decode([]byte(info.Text))
			if block == nil {
				s.error(w, errors.New("invalid certificate"))
				return
			}
			_, err := x509.ParseCertificate(block.Bytes)
			if err != nil {
				s.error(w, err)
				return
			}

			id := uuid.New().String()
			for x, cert := range s.Trust {
				if info.Text == cert.Text {
					id = x // existing certificate
					break
				}
			}
			s.Trust[id] = info

			w.WriteHeader(http.StatusCreated)
		}
	default:
		w.WriteHeader(http.StatusMethodNotAllowed)
	}
}

func (s *handler) libraryTrustedCertificatesID(w http.ResponseWriter, r *http.Request) {
	id := path.Base(r.URL.Path)
	cert, ok := s.Trust[id]
	if !ok {
		http.NotFound(w, r)
		return
	}

	switch r.Method {
	case http.MethodGet:
		StatusOK(w, &cert)
	case http.MethodDelete:
		delete(s.Trust, id)
	default:
		w.WriteHeader(http.StatusMethodNotAllowed)
	}
}

func (s *handler) vmID(w http.ResponseWriter, r *http.Request) {
	id := path.Base(r.URL.Path)

	switch r.Method {
	case http.MethodDelete:
		s.deleteVM(&types.ManagedObjectReference{Type: "VirtualMachine", Value: id})
	default:
		http.NotFound(w, r)
	}
}

func (s *handler) debugEcho(w http.ResponseWriter, r *http.Request) {
	r.Write(w)
}
