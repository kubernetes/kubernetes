/*
Copyright 2014 Google Inc. All rights reserved.

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

import "net/http"

type FakeStore struct {
	Session Session
	Err     error
	Saved   bool
	Wrapped bool
}

func (f *FakeStore) Get(req *http.Request, name string) (Session, error) {
	return f.Session, f.Err
}

func (f *FakeStore) Save(w http.ResponseWriter, req *http.Request) error {
	f.Saved = true
	return f.Err
}

func (f *FakeStore) Wrap(h http.Handler) http.Handler {
	f.Wrapped = true
	return h
}

type FakeSession struct {
	V map[interface{}]interface{}
}

func (f *FakeSession) Values() map[interface{}]interface{} {
	return f.V
}
