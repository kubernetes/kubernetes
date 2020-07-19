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
	"bytes"
	"errors"
	"fmt"
	"reflect"
	"sort"
	"sync/atomic"
	"time"

	"go.opencensus.io/metric/metricdata"
	"go.opencensus.io/stats"
	"go.opencensus.io/tag"
)

// View allows users to aggregate the recorded stats.Measurements.
// Views need to be passed to the Register function before data will be
// collected and sent to Exporters.
type View struct {
	Name        string // Name of View. Must be unique. If unset, will default to the name of the Measure.
	Description string // Description is a human-readable description for this view.

	// TagKeys are the tag keys describing the grouping of this view.
	// A single Row will be produced for each combination of associated tag values.
	TagKeys []tag.Key

	// Measure is a stats.Measure to aggregate in this view.
	Measure stats.Measure

	// Aggregation is the aggregation function to apply to the set of Measurements.
	Aggregation *Aggregation
}

// WithName returns a copy of the View with a new name. This is useful for
// renaming views to cope with limitations placed on metric names by various
// backends.
func (v *View) WithName(name string) *View {
	vNew := *v
	vNew.Name = name
	return &vNew
}

// same compares two views and returns true if they represent the same aggregation.
func (v *View) same(other *View) bool {
	if v == other {
		return true
	}
	if v == nil {
		return false
	}
	return reflect.DeepEqual(v.Aggregation, other.Aggregation) &&
		v.Measure.Name() == other.Measure.Name()
}

// ErrNegativeBucketBounds error returned if histogram contains negative bounds.
//
// Deprecated: this should not be public.
var ErrNegativeBucketBounds = errors.New("negative bucket bounds not supported")

// canonicalize canonicalizes v by setting explicit
// defaults for Name and Description and sorting the TagKeys
func (v *View) canonicalize() error {
	if v.Measure == nil {
		return fmt.Errorf("cannot register view %q: measure not set", v.Name)
	}
	if v.Aggregation == nil {
		return fmt.Errorf("cannot register view %q: aggregation not set", v.Name)
	}
	if v.Name == "" {
		v.Name = v.Measure.Name()
	}
	if v.Description == "" {
		v.Description = v.Measure.Description()
	}
	if err := checkViewName(v.Name); err != nil {
		return err
	}
	sort.Slice(v.TagKeys, func(i, j int) bool {
		return v.TagKeys[i].Name() < v.TagKeys[j].Name()
	})
	sort.Float64s(v.Aggregation.Buckets)
	for _, b := range v.Aggregation.Buckets {
		if b < 0 {
			return ErrNegativeBucketBounds
		}
	}
	// drop 0 bucket silently.
	v.Aggregation.Buckets = dropZeroBounds(v.Aggregation.Buckets...)

	return nil
}

func dropZeroBounds(bounds ...float64) []float64 {
	for i, bound := range bounds {
		if bound > 0 {
			return bounds[i:]
		}
	}
	return []float64{}
}

// viewInternal is the internal representation of a View.
type viewInternal struct {
	view             *View  // view is the canonicalized View definition associated with this view.
	subscribed       uint32 // 1 if someone is subscribed and data need to be exported, use atomic to access
	collector        *collector
	metricDescriptor *metricdata.Descriptor
}

func newViewInternal(v *View) (*viewInternal, error) {
	return &viewInternal{
		view:             v,
		collector:        &collector{make(map[string]AggregationData), v.Aggregation},
		metricDescriptor: viewToMetricDescriptor(v),
	}, nil
}

func (v *viewInternal) subscribe() {
	atomic.StoreUint32(&v.subscribed, 1)
}

func (v *viewInternal) unsubscribe() {
	atomic.StoreUint32(&v.subscribed, 0)
}

// isSubscribed returns true if the view is exporting
// data by subscription.
func (v *viewInternal) isSubscribed() bool {
	return atomic.LoadUint32(&v.subscribed) == 1
}

func (v *viewInternal) clearRows() {
	v.collector.clearRows()
}

func (v *viewInternal) collectedRows() []*Row {
	return v.collector.collectedRows(v.view.TagKeys)
}

func (v *viewInternal) addSample(m *tag.Map, val float64, attachments map[string]interface{}, t time.Time) {
	if !v.isSubscribed() {
		return
	}
	sig := string(encodeWithKeys(m, v.view.TagKeys))
	v.collector.addSample(sig, val, attachments, t)
}

// A Data is a set of rows about usage of the single measure associated
// with the given view. Each row is specific to a unique set of tags.
type Data struct {
	View       *View
	Start, End time.Time
	Rows       []*Row
}

// Row is the collected value for a specific set of key value pairs a.k.a tags.
type Row struct {
	Tags []tag.Tag
	Data AggregationData
}

func (r *Row) String() string {
	var buffer bytes.Buffer
	buffer.WriteString("{ ")
	buffer.WriteString("{ ")
	for _, t := range r.Tags {
		buffer.WriteString(fmt.Sprintf("{%v %v}", t.Key.Name(), t.Value))
	}
	buffer.WriteString(" }")
	buffer.WriteString(fmt.Sprintf("%v", r.Data))
	buffer.WriteString(" }")
	return buffer.String()
}

// Equal returns true if both rows are equal. Tags are expected to be ordered
// by the key name. Even if both rows have the same tags but the tags appear in
// different orders it will return false.
func (r *Row) Equal(other *Row) bool {
	if r == other {
		return true
	}
	return reflect.DeepEqual(r.Tags, other.Tags) && r.Data.equal(other.Data)
}

const maxNameLength = 255

// Returns true if the given string contains only printable characters.
func isPrintable(str string) bool {
	for _, r := range str {
		if !(r >= ' ' && r <= '~') {
			return false
		}
	}
	return true
}

func checkViewName(name string) error {
	if len(name) > maxNameLength {
		return fmt.Errorf("view name cannot be larger than %v", maxNameLength)
	}
	if !isPrintable(name) {
		return fmt.Errorf("view name needs to be an ASCII string")
	}
	return nil
}
