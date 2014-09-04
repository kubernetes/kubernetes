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

package etcdex

// Types of indexes that etcdex can maintain.

// Set holds a read-only set of strings with efficient membership checking.
// Membership is maintained by hidden goroutines.
type Set interface {
  Contains(key string) bool
}

// Map holds a read-only map[string]string.
// Membership is maintained by hidden goroutines.
type Map interface {
  Get(key string) string
}

// MultiMap holds a read-only map[string](map[string]string]).
// Membership is maintained by hidden goroutines.
type MultiMap interface {
  Get(key string) Set
}

type Client inteface {
  // IndexKeys creates, or returns a reference to an existing, Set of all the keys under etcdPathPfx. 
  // It also causes the membership of the Set to be maintained by goroutines that watch etcd.
  IndexKeys(etcdPathPfx string) *Set

  // IndexKeysAndValues creates, or returns a reference to an existing, Map of all the keys and values under etcdPathPfx. 
  // It also causes the membership of the Map to be maintained by goroutines that watch etcd.
  IndexKeysAndValues(etcdPathPfx string) *Map

  // IndexKeysAndSubValues creates, or returns a reference to an existing, Map of all the keys under etcdPathPfx,
  // and a part of each key's value.
  // It also causes the membership of the Map to be maintained by goroutines that watch etcd.
  // The value in the Map is a single field of the etcd value type, where the latter must be a JSON string.
  // IndexKeysAndSubValues is less CPU effiicent than IndexKeysAndValues since it has to decode JSON,
  // but more memory efficient since it discards parts of the value type that are not needed.
  IndexKeysAndSubValues(etcdPathPfx string, subValueProjectionExpression string) *Map

  // InvertedIndex creates, or returns a reference to an existing, MultiMap.
  // The Multimap's keys are unique values of property names of (JSON) etcd values.
  // The Multimap's value-Set contains all the etcd keys with that sub-value.
  InvertedIndex(etcdPathPfx string, subValueProjectionExpression string) *MulitMap
}

// New makes a new Client to etcd and allow for creation of, and holds, indexes.
// TODO: be able to dedup redundant Clients and/or Indexes at the go Package Level
 func NewConnection(p etcdConnectionParams) Connection

// TODO: make a transaction object, and have any index lookups done while that transaction is open
// cause the indexes state to be recorded, and then at commit time, do writes to etcd conditional 
// on the state of the indexes not having changed.


/**** Implementation ****/


type client struct {

  // collection of Indexes

  // connection to etcd, and watches on all relevant "dirs".
}

func (*client) IndexKeys(etcdPatchPfx string) *Set {
  // Create a type set.  

  // Start a goroutine to Watch this path prefix if not already watched.

  // Start a goroutine to subscribe to that Watch and to insert/delete changes into this particular set.

  // return pointer to set as Set interface.
}
// Similarly for other methods of Client ...

type set struct {
  data map[string]bool
}
// Similarly for Map and MultiMap ...

func (*set) Contains(key string) {
  // Acquire lock.
  // defer release lock.
  return data[key]
}
// Similarly for Map and MultiMap ...

// Alternate more-go-ish implementation:
// - actual storage (e.g. map[string]string) is private to an index object.
// - index object listens on channel for read/write requests and handles in series.
// - The Set/Map/MultiMap object returned to the user sends read requests over a that channel.

// TODO: transactions:
// This would appear to require that all etcd writers to tables requiring transactions should
// use some kind of synchronization.  That is beyond the scope of this sketch, but appears needed
// regardless of whether this sketch is expanded to a full implementation.
