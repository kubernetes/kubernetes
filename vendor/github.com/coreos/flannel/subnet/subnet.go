// Copyright 2015 flannel authors
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

package subnet

import (
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"strconv"
	"time"

	"github.com/coreos/flannel/pkg/ip"
	"golang.org/x/net/context"
)

var (
	ErrLeaseTaken  = errors.New("subnet: lease already taken")
	ErrNoMoreTries = errors.New("subnet: no more tries")
)

type LeaseAttrs struct {
	PublicIP    ip.IP4
	BackendType string          `json:",omitempty"`
	BackendData json.RawMessage `json:",omitempty"`
}

type Lease struct {
	Subnet     ip.IP4Net
	Attrs      LeaseAttrs
	Expiration time.Time

	asof uint64
}

func (l *Lease) Key() string {
	return MakeSubnetKey(l.Subnet)
}

type Reservation struct {
	Subnet   ip.IP4Net
	PublicIP ip.IP4
}

type (
	EventType int

	Event struct {
		Type    EventType `json:"type"`
		Lease   Lease     `json:"lease,omitempty"`
		Network string    `json:"network,omitempty"`
	}
)

const (
	EventAdded EventType = iota
	EventRemoved
)

type LeaseWatchResult struct {
	// Either Events or Snapshot will be set.  If Events is empty, it means
	// the cursor was out of range and Snapshot contains the current list
	// of items, even if empty.
	Events   []Event     `json:"events"`
	Snapshot []Lease     `json:"snapshot"`
	Cursor   interface{} `json:"cursor"`
}

type NetworkWatchResult struct {
	// Either Events or Snapshot will be set.  If Events is empty, it means
	// the cursor was out of range and Snapshot contains the current list
	// of items, even if empty.
	Events   []Event     `json:"events"`
	Snapshot []string    `json:"snapshot"`
	Cursor   interface{} `json:"cursor,omitempty"`
}

func (et EventType) MarshalJSON() ([]byte, error) {
	s := ""

	switch et {
	case EventAdded:
		s = "added"
	case EventRemoved:
		s = "removed"
	default:
		return nil, errors.New("bad event type")
	}
	return json.Marshal(s)
}

func (et *EventType) UnmarshalJSON(data []byte) error {
	switch string(data) {
	case "\"added\"":
		*et = EventAdded
	case "\"removed\"":
		*et = EventRemoved
	default:
		fmt.Println(string(data))
		return errors.New("bad event type")
	}

	return nil
}

func ParseSubnetKey(s string) *ip.IP4Net {
	if parts := subnetRegex.FindStringSubmatch(s); len(parts) == 3 {
		snIp := net.ParseIP(parts[1]).To4()
		prefixLen, err := strconv.ParseUint(parts[2], 10, 5)
		if snIp != nil && err == nil {
			return &ip.IP4Net{IP: ip.FromIP(snIp), PrefixLen: uint(prefixLen)}
		}
	}

	return nil
}

func MakeSubnetKey(sn ip.IP4Net) string {
	return sn.StringSep(".", "-")
}

type Manager interface {
	GetNetworkConfig(ctx context.Context, network string) (*Config, error)
	AcquireLease(ctx context.Context, network string, attrs *LeaseAttrs) (*Lease, error)
	RenewLease(ctx context.Context, network string, lease *Lease) error
	RevokeLease(ctx context.Context, network string, sn ip.IP4Net) error
	WatchLease(ctx context.Context, network string, sn ip.IP4Net, cursor interface{}) (LeaseWatchResult, error)
	WatchLeases(ctx context.Context, network string, cursor interface{}) (LeaseWatchResult, error)
	WatchNetworks(ctx context.Context, cursor interface{}) (NetworkWatchResult, error)

	AddReservation(ctx context.Context, network string, r *Reservation) error
	RemoveReservation(ctx context.Context, network string, subnet ip.IP4Net) error
	ListReservations(ctx context.Context, network string) ([]Reservation, error)
}
