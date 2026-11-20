//go:build linux

/*
Copyright The Kubernetes Authors.

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

package iptables

import (
	"context"
	"time"

	"k8s.io/klog/v2"
)

const (
	// Max time we wait for an iptables flush to complete after we notice it has started
	iptablesFlushTimeout = 5 * time.Second
	// How often we poll while waiting for an iptables flush to complete
	iptablesFlushPollTime = 100 * time.Millisecond
)

type monitor struct {
	ipt        *runner
	ctx        context.Context
	canary     Chain
	tables     []Table
	reloadFunc func()
	interval   time.Duration
}

// Monitor is part of Interface
func (ipt *runner) Monitor(ctx context.Context, canary Chain, tables []Table, reloadFunc func(), interval time.Duration) {
	mon := &monitor{ipt, ctx, canary, tables, reloadFunc, interval}
	mon.run()
}

func (mon *monitor) run() {
	defer mon.cleanupCanaries()
	for {
		mon.ensureCanaries()

		for {
			time.Sleep(mon.interval)
			if mon.isStopped() {
				return
			}
			if mon.primaryCanaryDeleted() {
				break
			}
		}

		klog.V(2).InfoS("IPTables canary deleted", "table", mon.tables[0], "chain", mon.canary)
		// Wait for the other canaries to be deleted too, so we don't
		// start reloading too soon.
		waitStart := time.Now()
		for !mon.allCanariesDeleted() && time.Since(waitStart) < iptablesFlushTimeout {
			time.Sleep(iptablesFlushPollTime)
		}
		if time.Since(waitStart) >= iptablesFlushTimeout {
			klog.InfoS("Inconsistent iptables state detected")
		}

		klog.V(2).InfoS("Reloading after iptables flush")
		mon.reloadFunc()
	}
}

func (mon *monitor) ensureCanaries() {
	for {
		if mon.isStopped() {
			return
		}
		failed := false
		for _, table := range mon.tables {
			if _, err := mon.ipt.EnsureChain(mon.ctx, table, mon.canary); err != nil {
				klog.ErrorS(err, "Could not set up iptables canary", "table", table, "chain", mon.canary)
				failed = true
				break
			}
		}
		if !failed {
			return
		}
		// Sleep and try again
		time.Sleep(mon.interval)
	}
}

func (mon *monitor) primaryCanaryDeleted() bool {
	exists, err := mon.ipt.ChainExists(mon.ctx, mon.tables[0], mon.canary)
	if isResourceError(err) {
		klog.ErrorS(err, "Could not check for iptables canary", "table", mon.tables[0], "chain", mon.canary)
		// If the check failed, then probably trying to reload right now would
		// fail too, so just assume the canary hasn't been deleted.
		return false
	}
	return !exists
}

func (mon *monitor) allCanariesDeleted() bool {
	for _, table := range mon.tables {
		if exists, err := mon.ipt.ChainExists(mon.ctx, table, mon.canary); exists || isResourceError(err) {
			return false
		}
	}
	return true
}

func (mon *monitor) cleanupCanaries() {
	for _, table := range mon.tables {
		// We intentionally don't use mon.ctx here, because it may be cancelled.
		_ = mon.ipt.DeleteChain(context.Background(), table, mon.canary)
	}
}

func (mon *monitor) isStopped() bool {
	return mon.ctx.Err() != nil
}
