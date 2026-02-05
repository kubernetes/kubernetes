// Copyright The Prometheus Authors
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

import (
	"bufio"
	"bytes"
	"fmt"

	"github.com/prometheus/procfs/internal/util"
)

const nfNetLinkQueueFormat = "%d %d %d %d %d %d %d %d %d"

// NFNetLinkQueue contains general information about netfilter queues found in /proc/net/netfilter/nfnetlink_queue.
type NFNetLinkQueue struct {
	// id of the queue
	QueueID uint
	// pid of process handling the queue
	PeerPID uint
	// number of packets waiting for a decision
	QueueTotal uint
	// indicate how userspace receive packets
	CopyMode uint
	// size of copy
	CopyRange uint
	// number of items dropped by the kernel because too many packets were waiting a decision.
	// It queue_total is superior to queue_max_len (1024 per default) the packets are dropped.
	QueueDropped uint
	// number of packets dropped by userspace (due to kernel send failure on the netlink socket)
	QueueUserDropped uint
	// sequence number of packets queued. It gives a correct approximation of the number of queued packets.
	SequenceID uint
	// internal value (number of entity using the queue)
	Use uint
}

// NFNetLinkQueue returns information about current state of netfilter queues.
func (fs FS) NFNetLinkQueue() ([]NFNetLinkQueue, error) {
	data, err := util.ReadFileNoStat(fs.proc.Path("net/netfilter/nfnetlink_queue"))
	if err != nil {
		return nil, err
	}

	queue := []NFNetLinkQueue{}
	if len(data) == 0 {
		return queue, nil
	}

	scanner := bufio.NewScanner(bytes.NewReader(data))
	for scanner.Scan() {
		line := scanner.Text()
		nFNetLinkQueue, err := parseNFNetLinkQueueLine(line)
		if err != nil {
			return nil, err
		}
		queue = append(queue, *nFNetLinkQueue)
	}
	return queue, nil
}

// parseNFNetLinkQueueLine parses each line of the /proc/net/netfilter/nfnetlink_queue file.
func parseNFNetLinkQueueLine(line string) (*NFNetLinkQueue, error) {
	nFNetLinkQueue := NFNetLinkQueue{}
	_, err := fmt.Sscanf(
		line, nfNetLinkQueueFormat,
		&nFNetLinkQueue.QueueID, &nFNetLinkQueue.PeerPID, &nFNetLinkQueue.QueueTotal, &nFNetLinkQueue.CopyMode,
		&nFNetLinkQueue.CopyRange, &nFNetLinkQueue.QueueDropped, &nFNetLinkQueue.QueueUserDropped, &nFNetLinkQueue.SequenceID, &nFNetLinkQueue.Use,
	)
	if err != nil {
		return nil, err
	}
	return &nFNetLinkQueue, nil
}
