// Copyright 2015 CoreOS, Inc.
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

package ip

import (
	"io/ioutil"
)

func EnableIP4Forward() error {
	return echo1("/proc/sys/net/ipv4/ip_forward")
}

func EnableIP6Forward() error {
	return echo1("/proc/sys/net/ipv6/conf/all/forwarding")
}

func echo1(f string) error {
	return ioutil.WriteFile(f, []byte("1"), 0644)
}
