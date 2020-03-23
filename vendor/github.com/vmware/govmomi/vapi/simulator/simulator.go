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

package simulator

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"path"
	"reflect"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/vmware/govmomi/vapi/internal"
	"github.com/vmware/govmomi/vapi/tags"
	vim "github.com/vmware/govmomi/vim25/types"
)

type session struct {
	User         string    `json:"user"`
	Created      time.Time `json:"created_time"`
	LastAccessed time.Time `json:"last_accessed_time"`
}

type handler struct {
	*http.ServeMux
	sync.Mutex
	Category    map[string]*tags.Category
	Tag         map[string]*tags.Tag
	Association map[string]map[internal.AssociatedObject]bool
	Session     map[string]*session
}

// New creates a vAPI simulator.
func New(u *url.URL, settings []vim.BaseOptionValue) (string, http.Handler) {
	s := &handler{
		ServeMux:    http.NewServeMux(),
		Category:    make(map[string]*tags.Category),
		Tag:         make(map[string]*tags.Tag),
		Association: make(map[string]map[internal.AssociatedObject]bool),
		Session:     make(map[string]*session),
	}

	handlers := []struct {
		p string
		m http.HandlerFunc
	}{
		{internal.SessionPath, s.session},
		{internal.CategoryPath, s.category},
		{internal.CategoryPath + "/", s.categoryID},
		{internal.TagPath, s.tag},
		{internal.TagPath + "/", s.tagID},
		{internal.AssociationPath, s.association},
		{internal.AssociationPath + "/", s.associationID},
	}

	for i := range handlers {
		h := handlers[i]
		s.HandleFunc(internal.Path+h.p, func(w http.ResponseWriter, r *http.Request) {
			s.Lock()
			defer s.Unlock()

			if !s.isAuthorized(r) {
				w.WriteHeader(http.StatusUnauthorized)
				return
			}

			h.m(w, r)
		})
	}

	return internal.Path + "/", s
}

func (s *handler) isAuthorized(r *http.Request) bool {
	if r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, internal.SessionPath) {
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
	}
	return ok
}

func (s *handler) hasAuthorization(r *http.Request) (string, bool) {
	u, p, ok := r.BasicAuth()
	if ok { // user+pass auth
		if u == "" || p == "" {
			return u, false
		}
		return u, true
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

// ok responds with http.StatusOK and json encodes val if given.
func (s *handler) ok(w http.ResponseWriter, val ...interface{}) {
	w.WriteHeader(http.StatusOK)

	if len(val) == 0 {
		return
	}

	err := json.NewEncoder(w).Encode(struct {
		Value interface{} `json:"value,omitempty"`
	}{
		val[0],
	})

	if err != nil {
		log.Panic(err)
	}
}

func (s *handler) fail(w http.ResponseWriter, kind string) {
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

// ServeHTTP handles vAPI requests.
func (s *handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodPost, http.MethodDelete, http.MethodGet, http.MethodPatch:
	default:
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	h, _ := s.Handler(r)
	h.ServeHTTP(w, r)
}

func (s *handler) decode(r *http.Request, w http.ResponseWriter, val interface{}) bool {
	defer r.Body.Close()
	err := json.NewDecoder(r.Body).Decode(val)
	if err != nil {
		log.Printf("%s %s: %s", r.Method, r.RequestURI, err)
		w.WriteHeader(http.StatusBadRequest)
		return false
	}
	return true
}

func (s *handler) session(w http.ResponseWriter, r *http.Request) {
	id := r.Header.Get(internal.SessionCookieName)

	switch r.Method {
	case http.MethodPost:
		user, ok := s.hasAuthorization(r)
		if !ok {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		id = uuid.New().String()
		now := time.Now()
		s.Session[id] = &session{user, now, now}
		http.SetCookie(w, &http.Cookie{
			Name:  internal.SessionCookieName,
			Value: id,
		})
		s.ok(w, id)
	case http.MethodDelete:
		delete(s.Session, id)
		s.ok(w)
	case http.MethodGet:
		s.ok(w, s.Session[id])
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
					s.fail(w, "com.vmware.vapi.std.errors.already_exists")
					return
				}
			}
			id := newID("Category")
			spec.Category.ID = id
			s.Category[id] = &spec.Category
			s.ok(w, id)
		}
	case http.MethodGet:
		var ids []string
		for id := range s.Category {
			ids = append(ids, id)
		}

		s.ok(w, ids)
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
		s.ok(w)
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
					s.fail(w, "com.vmware.vapi.std.errors.invalid_argument")
					return
				}
			}
			o.Patch(&spec.Category)
			s.ok(w)
		}
	case http.MethodGet:
		s.ok(w, o)
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
				if tag.Name == spec.Tag.Name {
					s.fail(w, "com.vmware.vapi.std.errors.already_exists")
					return
				}
			}
			id := newID("Tag")
			spec.Tag.ID = id
			s.Tag[id] = &spec.Tag
			s.Association[id] = make(map[internal.AssociatedObject]bool)
			s.ok(w, id)
		}
	case http.MethodGet:
		var ids []string
		for id := range s.Tag {
			ids = append(ids, id)
		}
		s.ok(w, ids)
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
		s.ok(w, ids)
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
		s.ok(w)
	case http.MethodPatch:
		var spec struct {
			Tag tags.Tag `json:"update_spec"`
		}
		if s.decode(r, w, &spec) {
			o.Patch(&spec.Tag)
			s.ok(w)
		}
	case http.MethodGet:
		s.ok(w, o)
	}
}

func (s *handler) association(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	var spec internal.Association
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
		s.ok(w, ids)
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
	if !s.decode(r, w, &spec) {
		return
	}

	switch s.action(r) {
	case "attach":
		s.Association[id][*spec.ObjectID] = true
		s.ok(w)
	case "detach":
		delete(s.Association[id], *spec.ObjectID)
		s.ok(w)
	case "list-attached-objects":
		var ids []internal.AssociatedObject
		for id := range s.Association[id] {
			ids = append(ids, id)
		}
		s.ok(w, ids)
	}
}
