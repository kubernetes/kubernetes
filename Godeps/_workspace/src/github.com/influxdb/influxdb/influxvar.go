package influxdb

import (
	"expvar"
	"sync"
)

var expvarMu sync.Mutex

// NewStatistics returns an expvar-based map with the given key. Within that map
// is another map. Within there "name" is the Measurement name, "tags" are the tags,
// and values are placed at the key "values".
func NewStatistics(key, name string, tags map[string]string) *expvar.Map {
	expvarMu.Lock()
	defer expvarMu.Unlock()

	// Add expvar for this service.
	var v expvar.Var
	if v = expvar.Get(key); v == nil {
		v = expvar.NewMap(key)
	}
	m := v.(*expvar.Map)

	// Set the name
	nameVar := &expvar.String{}
	nameVar.Set(name)
	m.Set("name", nameVar)

	// Set the tags
	tagsVar := &expvar.Map{}
	tagsVar.Init()
	for k, v := range tags {
		value := &expvar.String{}
		value.Set(v)
		tagsVar.Set(k, value)
	}
	m.Set("tags", tagsVar)

	// Create and set the values entry used for actual stats.
	statMap := &expvar.Map{}
	statMap.Init()
	m.Set("values", statMap)

	return statMap
}
