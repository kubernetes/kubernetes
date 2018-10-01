// Copyright 2017, OpenCensus Authors
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
//

package view

import (
	"errors"
	"fmt"
	"strings"
	"time"

	"go.opencensus.io/stats"
	"go.opencensus.io/stats/internal"
	"go.opencensus.io/tag"
)

type command interface {
	handleCommand(w *worker)
}

// getViewByNameReq is the command to get a view given its name.
type getViewByNameReq struct {
	name string
	c    chan *getViewByNameResp
}

type getViewByNameResp struct {
	v *View
}

func (cmd *getViewByNameReq) handleCommand(w *worker) {
	v := w.views[cmd.name]
	if v == nil {
		cmd.c <- &getViewByNameResp{nil}
		return
	}
	cmd.c <- &getViewByNameResp{v.view}
}

// registerViewReq is the command to register a view.
type registerViewReq struct {
	views []*View
	err   chan error
}

func (cmd *registerViewReq) handleCommand(w *worker) {
	var errstr []string
	for _, view := range cmd.views {
		vi, err := w.tryRegisterView(view)
		if err != nil {
			errstr = append(errstr, fmt.Sprintf("%s: %v", view.Name, err))
			continue
		}
		internal.SubscriptionReporter(view.Measure.Name())
		vi.subscribe()
	}
	if len(errstr) > 0 {
		cmd.err <- errors.New(strings.Join(errstr, "\n"))
	} else {
		cmd.err <- nil
	}
}

// unregisterFromViewReq is the command to unregister to a view. Has no
// impact on the data collection for client that are pulling data from the
// library.
type unregisterFromViewReq struct {
	views []string
	done  chan struct{}
}

func (cmd *unregisterFromViewReq) handleCommand(w *worker) {
	for _, name := range cmd.views {
		vi, ok := w.views[name]
		if !ok {
			continue
		}

		// Report pending data for this view before removing it.
		w.reportView(vi, time.Now())

		vi.unsubscribe()
		if !vi.isSubscribed() {
			// this was the last subscription and view is not collecting anymore.
			// The collected data can be cleared.
			vi.clearRows()
		}
		delete(w.views, name)
	}
	cmd.done <- struct{}{}
}

// retrieveDataReq is the command to retrieve data for a view.
type retrieveDataReq struct {
	now time.Time
	v   string
	c   chan *retrieveDataResp
}

type retrieveDataResp struct {
	rows []*Row
	err  error
}

func (cmd *retrieveDataReq) handleCommand(w *worker) {
	vi, ok := w.views[cmd.v]
	if !ok {
		cmd.c <- &retrieveDataResp{
			nil,
			fmt.Errorf("cannot retrieve data; view %q is not registered", cmd.v),
		}
		return
	}

	if !vi.isSubscribed() {
		cmd.c <- &retrieveDataResp{
			nil,
			fmt.Errorf("cannot retrieve data; view %q has no subscriptions or collection is not forcibly started", cmd.v),
		}
		return
	}
	cmd.c <- &retrieveDataResp{
		vi.collectedRows(),
		nil,
	}
}

// recordReq is the command to record data related to multiple measures
// at once.
type recordReq struct {
	tm *tag.Map
	ms []stats.Measurement
}

func (cmd *recordReq) handleCommand(w *worker) {
	for _, m := range cmd.ms {
		if (m == stats.Measurement{}) { // not registered
			continue
		}
		ref := w.getMeasureRef(m.Measure().Name())
		for v := range ref.views {
			v.addSample(cmd.tm, m.Value())
		}
	}
}

// setReportingPeriodReq is the command to modify the duration between
// reporting the collected data to the registered clients.
type setReportingPeriodReq struct {
	d time.Duration
	c chan bool
}

func (cmd *setReportingPeriodReq) handleCommand(w *worker) {
	w.timer.Stop()
	if cmd.d <= 0 {
		w.timer = time.NewTicker(defaultReportingDuration)
	} else {
		w.timer = time.NewTicker(cmd.d)
	}
	cmd.c <- true
}
