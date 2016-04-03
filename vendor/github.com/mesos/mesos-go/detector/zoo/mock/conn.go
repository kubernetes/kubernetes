/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package mock

import (
	"github.com/samuel/go-zookeeper/zk"
	"github.com/stretchr/testify/mock"
)

// Impersontates a zk.Connection
// It implements interface Connector
type Connector struct {
	mock.Mock
}

func NewConnector() *Connector {
	return new(Connector)
}

func (conn *Connector) Close() {
	conn.Called()
}

func (conn *Connector) ChildrenW(path string) ([]string, *zk.Stat, <-chan zk.Event, error) {
	args := conn.Called(path)
	var (
		arg0 []string
		arg1 *zk.Stat
		arg2 <-chan zk.Event
	)
	if args.Get(0) != nil {
		arg0 = args.Get(0).([]string)
	}
	if args.Get(1) != nil {
		arg1 = args.Get(1).(*zk.Stat)
	}
	if args.Get(2) != nil {
		arg2 = args.Get(2).(<-chan zk.Event)
	}
	return arg0, arg1, arg2, args.Error(3)
}

func (conn *Connector) Children(path string) ([]string, *zk.Stat, error) {
	args := conn.Called(path)
	return args.Get(0).([]string),
		args.Get(1).(*zk.Stat),
		args.Error(2)
}

func (conn *Connector) Get(path string) ([]byte, *zk.Stat, error) {
	args := conn.Called(path)
	return args.Get(0).([]byte),
		args.Get(1).(*zk.Stat),
		args.Error(2)
}
