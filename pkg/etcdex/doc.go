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

// Package etcdex provides a read-only "coherent" in-memory content-addressible cache of etcd content.
//
// It assumes that etcd is being used according to the following extended analogy:
// Etcd concept                 : Relational Database Concept ::
//
// tree rooted at "dir"         : Table ::
// etcd file part after "dir"   : Primary Key ::
// etcd value (must be JSON)    : Row (except Primary Key) ::
// union of all fully qualified
// JSON property names of etcd
// values in "dir"              : Columns of a Table
//
// It provides for the creation of "indexes" of "tables" stored in etcd.
// The name is a contraction of etcd and index.
// It is not intended to be a completehttp://en.wikipedia.org/wiki/Object-relational_mapping
// It watches etcd to keep its indexes up to date.

package etcdex
