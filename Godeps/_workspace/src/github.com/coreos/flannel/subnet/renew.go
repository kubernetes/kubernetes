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

package subnet

import (
	"time"

	"github.com/coreos/flannel/Godeps/_workspace/src/golang.org/x/net/context"
	"log"
)

const (
	renewMargin = time.Hour
)

func LeaseRenewer(ctx context.Context, m Manager, network string, lease *Lease) {
	dur := lease.Expiration.Sub(time.Now()) - renewMargin

	for {
		select {
		case <-time.After(dur):
			err := m.RenewLease(ctx, network, lease)
			if err != nil {
				log.Printf("Error renewing lease (trying again in 1 min): ", err)
				dur = time.Minute
				continue
			}

			log.Printf("Lease renewed, new expiration: ", lease.Expiration)
			dur = lease.Expiration.Sub(time.Now()) - renewMargin

		case <-ctx.Done():
			return
		}
	}
}
