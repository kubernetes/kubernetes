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
	// NetTCP represents the contents of /proc/net/tcp{,6} file without the header.
	NetTCP []*netIPSocketLine

	// NetTCPSummary provides already computed values like the total queue lengths or
	// the total number of used sockets. In contrast to NetTCP it does not collect
	// the parsed lines into a slice.
	NetTCPSummary NetIPSocketSummary
)

// NetTCP returns the IPv4 kernel/networking statistics for TCP datagrams
// read from /proc/net/tcp.
// Deprecated: Use github.com/mdlayher/netlink#Conn (with syscall.AF_INET) instead.
func (fs FS) NetTCP() (NetTCP, error) {
	return newNetTCP(fs.proc.Path("net/tcp"))
}

// NetTCP6 returns the IPv6 kernel/networking statistics for TCP datagrams
// read from /proc/net/tcp6.
// Deprecated: Use github.com/mdlayher/netlink#Conn (with syscall.AF_INET6) instead.
func (fs FS) NetTCP6() (NetTCP, error) {
	return newNetTCP(fs.proc.Path("net/tcp6"))
}

// NetTCPSummary returns already computed statistics like the total queue lengths
// for TCP datagrams read from /proc/net/tcp.
// Deprecated: Use github.com/mdlayher/netlink#Conn (with syscall.AF_INET) instead.
func (fs FS) NetTCPSummary() (*NetTCPSummary, error) {
	return newNetTCPSummary(fs.proc.Path("net/tcp"))
}

// NetTCP6Summary returns already computed statistics like the total queue lengths
// for TCP datagrams read from /proc/net/tcp6.
// Deprecated: Use github.com/mdlayher/netlink#Conn (with syscall.AF_INET6) instead.
func (fs FS) NetTCP6Summary() (*NetTCPSummary, error) {
	return newNetTCPSummary(fs.proc.Path("net/tcp6"))
}

// newNetTCP creates a new NetTCP{,6} from the contents of the given file.
func newNetTCP(file string) (NetTCP, error) {
	n, err := newNetIPSocket(file)
	n1 := NetTCP(n)
	return n1, err
}

func newNetTCPSummary(file string) (*NetTCPSummary, error) {
	n, err := newNetIPSocketSummary(file)
	if n == nil {
		return nil, err
	}
	n1 := NetTCPSummary(*n)
	return &n1, err
}
