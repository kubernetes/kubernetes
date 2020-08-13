// Copyright 2018 The etcd Authors
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

package balancer

import (
	"fmt"
	"net/url"
	"sort"
	"sync/atomic"
	"time"

	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/resolver"
)

func scToString(sc balancer.SubConn) string {
	return fmt.Sprintf("%p", sc)
}

func scsToStrings(scs map[balancer.SubConn]resolver.Address) (ss []string) {
	ss = make([]string, 0, len(scs))
	for sc, a := range scs {
		ss = append(ss, fmt.Sprintf("%s (%s)", a.Addr, scToString(sc)))
	}
	sort.Strings(ss)
	return ss
}

func addrsToStrings(addrs []resolver.Address) (ss []string) {
	ss = make([]string, len(addrs))
	for i := range addrs {
		ss[i] = addrs[i].Addr
	}
	sort.Strings(ss)
	return ss
}

func epsToAddrs(eps ...string) (addrs []resolver.Address) {
	addrs = make([]resolver.Address, 0, len(eps))
	for _, ep := range eps {
		u, err := url.Parse(ep)
		if err != nil {
			addrs = append(addrs, resolver.Address{Addr: ep, Type: resolver.Backend})
			continue
		}
		addrs = append(addrs, resolver.Address{Addr: u.Host, Type: resolver.Backend})
	}
	return addrs
}

var genN = new(uint32)

func genName() string {
	now := time.Now().UnixNano()
	return fmt.Sprintf("%X%X", now, atomic.AddUint32(genN, 1))
}
