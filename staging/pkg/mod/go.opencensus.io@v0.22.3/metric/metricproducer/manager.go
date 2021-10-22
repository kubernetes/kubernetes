// Copyright 2019, OpenCensus Authors
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

package metricproducer

import (
	"sync"
)

// Manager maintains a list of active producers. Producers can register
// with the manager to allow readers to read all metrics provided by them.
// Readers can retrieve all producers registered with the manager,
// read metrics from the producers and export them.
type Manager struct {
	mu        sync.RWMutex
	producers map[Producer]struct{}
}

var prodMgr *Manager
var once sync.Once

// GlobalManager is a single instance of producer manager
// that is used by all producers and all readers.
func GlobalManager() *Manager {
	once.Do(func() {
		prodMgr = &Manager{}
		prodMgr.producers = make(map[Producer]struct{})
	})
	return prodMgr
}

// AddProducer adds the producer to the Manager if it is not already present.
func (pm *Manager) AddProducer(producer Producer) {
	if producer == nil {
		return
	}
	pm.mu.Lock()
	defer pm.mu.Unlock()
	pm.producers[producer] = struct{}{}
}

// DeleteProducer deletes the producer from the Manager if it is present.
func (pm *Manager) DeleteProducer(producer Producer) {
	if producer == nil {
		return
	}
	pm.mu.Lock()
	defer pm.mu.Unlock()
	delete(pm.producers, producer)
}

// GetAll returns a slice of all producer currently registered with
// the Manager. For each call it generates a new slice. The slice
// should not be cached as registration may change at any time. It is
// typically called periodically by exporter to read metrics from
// the producers.
func (pm *Manager) GetAll() []Producer {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	producers := make([]Producer, len(pm.producers))
	i := 0
	for producer := range pm.producers {
		producers[i] = producer
		i++
	}
	return producers
}
