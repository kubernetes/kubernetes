// Copyright 2015 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package auth

import (
	"context"
	"encoding/json"
	"path"

	etcderr "github.com/coreos/etcd/error"
	"github.com/coreos/etcd/etcdserver"
	"github.com/coreos/etcd/etcdserver/etcdserverpb"
)

func (s *store) ensureAuthDirectories() error {
	if s.ensuredOnce {
		return nil
	}
	for _, res := range []string{StorePermsPrefix, StorePermsPrefix + "/users/", StorePermsPrefix + "/roles/"} {
		ctx, cancel := context.WithTimeout(context.Background(), s.timeout)
		defer cancel()
		pe := false
		rr := etcdserverpb.Request{
			Method:    "PUT",
			Path:      res,
			Dir:       true,
			PrevExist: &pe,
		}
		_, err := s.server.Do(ctx, rr)
		if err != nil {
			if e, ok := err.(*etcderr.Error); ok {
				if e.ErrorCode == etcderr.EcodeNodeExist {
					continue
				}
			}
			plog.Errorf("failed to create auth directories in the store (%v)", err)
			return err
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), s.timeout)
	defer cancel()
	pe := false
	rr := etcdserverpb.Request{
		Method:    "PUT",
		Path:      StorePermsPrefix + "/enabled",
		Val:       "false",
		PrevExist: &pe,
	}
	_, err := s.server.Do(ctx, rr)
	if err != nil {
		if e, ok := err.(*etcderr.Error); ok {
			if e.ErrorCode == etcderr.EcodeNodeExist {
				s.ensuredOnce = true
				return nil
			}
		}
		return err
	}
	s.ensuredOnce = true
	return nil
}

func (s *store) enableAuth() error {
	_, err := s.updateResource("/enabled", true)
	return err
}
func (s *store) disableAuth() error {
	_, err := s.updateResource("/enabled", false)
	return err
}

func (s *store) detectAuth() bool {
	if s.server == nil {
		return false
	}
	value, err := s.requestResource("/enabled", false, false)
	if err != nil {
		if e, ok := err.(*etcderr.Error); ok {
			if e.ErrorCode == etcderr.EcodeKeyNotFound {
				return false
			}
		}
		plog.Errorf("failed to detect auth settings (%s)", err)
		return false
	}

	var u bool
	err = json.Unmarshal([]byte(*value.Event.Node.Value), &u)
	if err != nil {
		plog.Errorf("internal bookkeeping value for enabled isn't valid JSON (%v)", err)
		return false
	}
	return u
}

func (s *store) requestResource(res string, dir, quorum bool) (etcdserver.Response, error) {
	ctx, cancel := context.WithTimeout(context.Background(), s.timeout)
	defer cancel()
	p := path.Join(StorePermsPrefix, res)
	method := "GET"
	if quorum {
		method = "QGET"
	}
	rr := etcdserverpb.Request{
		Method: method,
		Path:   p,
		Dir:    dir,
	}
	return s.server.Do(ctx, rr)
}

func (s *store) updateResource(res string, value interface{}) (etcdserver.Response, error) {
	return s.setResource(res, value, true)
}
func (s *store) createResource(res string, value interface{}) (etcdserver.Response, error) {
	return s.setResource(res, value, false)
}
func (s *store) setResource(res string, value interface{}, prevexist bool) (etcdserver.Response, error) {
	err := s.ensureAuthDirectories()
	if err != nil {
		return etcdserver.Response{}, err
	}
	ctx, cancel := context.WithTimeout(context.Background(), s.timeout)
	defer cancel()
	data, err := json.Marshal(value)
	if err != nil {
		return etcdserver.Response{}, err
	}
	p := path.Join(StorePermsPrefix, res)
	rr := etcdserverpb.Request{
		Method:    "PUT",
		Path:      p,
		Val:       string(data),
		PrevExist: &prevexist,
	}
	return s.server.Do(ctx, rr)
}

func (s *store) deleteResource(res string) (etcdserver.Response, error) {
	err := s.ensureAuthDirectories()
	if err != nil {
		return etcdserver.Response{}, err
	}
	ctx, cancel := context.WithTimeout(context.Background(), s.timeout)
	defer cancel()
	pex := true
	p := path.Join(StorePermsPrefix, res)
	rr := etcdserverpb.Request{
		Method:    "DELETE",
		Path:      p,
		PrevExist: &pex,
	}
	return s.server.Do(ctx, rr)
}
