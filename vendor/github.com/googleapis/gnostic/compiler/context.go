// Copyright 2017 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package compiler

// Context contains state of the compiler as it traverses a document.
type Context struct {
	Parent            *Context
	Name              string
	ExtensionHandlers *[]ExtensionHandler
}

// NewContextWithExtensions returns a new object representing the compiler state
func NewContextWithExtensions(name string, parent *Context, extensionHandlers *[]ExtensionHandler) *Context {
	return &Context{Name: name, Parent: parent, ExtensionHandlers: extensionHandlers}
}

// NewContext returns a new object representing the compiler state
func NewContext(name string, parent *Context) *Context {
	if parent != nil {
		return &Context{Name: name, Parent: parent, ExtensionHandlers: parent.ExtensionHandlers}
	}
	return &Context{Name: name, Parent: parent, ExtensionHandlers: nil}
}

// Description returns a text description of the compiler state
func (context *Context) Description() string {
	if context.Parent != nil {
		return context.Parent.Description() + "." + context.Name
	}
	return context.Name
}
