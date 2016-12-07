package watch

import (
	"fmt"
	"log"
	"os"
	"reflect"
	"time"

	consulapi "github.com/hashicorp/consul/api"
)

const (
	// retryInterval is the base retry value
	retryInterval = 5 * time.Second

	// maximum back off time, this is to prevent
	// exponential runaway
	maxBackoffTime = 180 * time.Second
)

// Run is used to run a watch plan
func (p *WatchPlan) Run(address string) error {
	// Setup the client
	p.address = address
	conf := consulapi.DefaultConfig()
	conf.Address = address
	conf.Datacenter = p.Datacenter
	conf.Token = p.Token
	client, err := consulapi.NewClient(conf)
	if err != nil {
		return fmt.Errorf("Failed to connect to agent: %v", err)
	}
	p.client = client

	// Create the logger
	output := p.LogOutput
	if output == nil {
		output = os.Stderr
	}
	logger := log.New(output, "", log.LstdFlags)

	// Loop until we are canceled
	failures := 0
OUTER:
	for !p.shouldStop() {
		// Invoke the handler
		index, result, err := p.Func(p)

		// Check if we should terminate since the function
		// could have blocked for a while
		if p.shouldStop() {
			break
		}

		// Handle an error in the watch function
		if err != nil {
			// Perform an exponential backoff
			failures++
			retry := retryInterval * time.Duration(failures*failures)
			if retry > maxBackoffTime {
				retry = maxBackoffTime
			}
			logger.Printf("consul.watch: Watch (type: %s) errored: %v, retry in %v",
				p.Type, err, retry)
			select {
			case <-time.After(retry):
				continue OUTER
			case <-p.stopCh:
				return nil
			}
		}

		// Clear the failures
		failures = 0

		// If the index is unchanged do nothing
		if index == p.lastIndex {
			continue
		}

		// Update the index, look for change
		oldIndex := p.lastIndex
		p.lastIndex = index
		if oldIndex != 0 && reflect.DeepEqual(p.lastResult, result) {
			continue
		}

		// Handle the updated result
		p.lastResult = result
		if p.Handler != nil {
			p.Handler(index, result)
		}
	}
	return nil
}

// Stop is used to stop running the watch plan
func (p *WatchPlan) Stop() {
	p.stopLock.Lock()
	defer p.stopLock.Unlock()
	if p.stop {
		return
	}
	p.stop = true
	close(p.stopCh)
}

func (p *WatchPlan) shouldStop() bool {
	select {
	case <-p.stopCh:
		return true
	default:
		return false
	}
}
