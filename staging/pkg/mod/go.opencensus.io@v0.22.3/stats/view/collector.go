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
	"sort"
	"time"

	"go.opencensus.io/internal/tagencoding"
	"go.opencensus.io/tag"
)

type collector struct {
	// signatures holds the aggregations values for each unique tag signature
	// (values for all keys) to its aggregator.
	signatures map[string]AggregationData
	// Aggregation is the description of the aggregation to perform for this
	// view.
	a *Aggregation
}

func (c *collector) addSample(s string, v float64, attachments map[string]interface{}, t time.Time) {
	aggregator, ok := c.signatures[s]
	if !ok {
		aggregator = c.a.newData()
		c.signatures[s] = aggregator
	}
	aggregator.addSample(v, attachments, t)
}

// collectRows returns a snapshot of the collected Row values.
func (c *collector) collectedRows(keys []tag.Key) []*Row {
	rows := make([]*Row, 0, len(c.signatures))
	for sig, aggregator := range c.signatures {
		tags := decodeTags([]byte(sig), keys)
		row := &Row{Tags: tags, Data: aggregator.clone()}
		rows = append(rows, row)
	}
	return rows
}

func (c *collector) clearRows() {
	c.signatures = make(map[string]AggregationData)
}

// encodeWithKeys encodes the map by using values
// only associated with the keys provided.
func encodeWithKeys(m *tag.Map, keys []tag.Key) []byte {
	vb := &tagencoding.Values{
		Buffer: make([]byte, len(keys)),
	}
	for _, k := range keys {
		v, _ := m.Value(k)
		vb.WriteValue([]byte(v))
	}
	return vb.Bytes()
}

// decodeTags decodes tags from the buffer and
// orders them by the keys.
func decodeTags(buf []byte, keys []tag.Key) []tag.Tag {
	vb := &tagencoding.Values{Buffer: buf}
	var tags []tag.Tag
	for _, k := range keys {
		v := vb.ReadValue()
		if v != nil {
			tags = append(tags, tag.Tag{Key: k, Value: string(v)})
		}
	}
	vb.ReadIndex = 0
	sort.Slice(tags, func(i, j int) bool { return tags[i].Key.Name() < tags[j].Key.Name() })
	return tags
}
