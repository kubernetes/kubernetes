// Copyright 2020 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package procfs

type (
	// NetUDP represents the contents of /proc/net/udp{,6} file without the header.
	NetUDP []*netIPSocketLine

	// NetUDPSummary provides already computed values like the total queue lengths or
	// the total number of used sockets. In contrast to NetUDP it does not collect
	// the parsed lines into a slice.
	NetUDPSummary NetIPSocketSummary
)

// NetUDP returns the IPv4 kernel/networking statistics for UDP datagrams
// read from /proc/net/udp.
func (fs FS) NetUDP() (NetUDP, error) {
	return newNetUDP(fs.proc.Path("net/udp"))
}

// NetUDP6 returns the IPv6 kernel/networking statistics for UDP datagrams
// read from /proc/net/udp6.
func (fs FS) NetUDP6() (NetUDP, error) {
	return newNetUDP(fs.proc.Path("net/udp6"))
}

// NetUDPSummary returns already computed statistics like the total queue lengths
// for UDP datagrams read from /proc/net/udp.
func (fs FS) NetUDPSummary() (*NetUDPSummary, error) {
	return newNetUDPSummary(fs.proc.Path("net/udp"))
}

// NetUDP6Summary returns already computed statistics like the total queue lengths
// for UDP datagrams read from /proc/net/udp6.
func (fs FS) NetUDP6Summary() (*NetUDPSummary, error) {
	return newNetUDPSummary(fs.proc.Path("net/udp6"))
}

// newNetUDP creates a new NetUDP{,6} from the contents of the given file.
func newNetUDP(file string) (NetUDP, error) {
	n, err := newNetIPSocket(file)
	n1 := NetUDP(n)
	return n1, err
}

func newNetUDPSummary(file string) (*NetUDPSummary, error) {
	n, err := newNetIPSocketSummary(file)
	if n == nil {
		return nil, err
	}
	n1 := NetUDPSummary(*n)
	return &n1, err
}
