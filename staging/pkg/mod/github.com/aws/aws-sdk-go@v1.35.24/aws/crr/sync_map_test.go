package crr

import (
	"reflect"
	"testing"
)

func TestRangeDelete(t *testing.T) {
	m := newSyncMap()
	for i := 0; i < 10; i++ {
		m.Store(i, i*10)
	}

	m.Range(func(key, value interface{}) bool {
		m.Delete(key)
		return true
	})

	expectedMap := map[interface{}]interface{}{}
	actualMap := map[interface{}]interface{}{}
	m.Range(func(key, value interface{}) bool {
		actualMap[key] = value
		return true
	})

	if e, a := len(expectedMap), len(actualMap); e != a {
		t.Errorf("expected map size %d, but received %d", e, a)
	}

	if e, a := expectedMap, actualMap; !reflect.DeepEqual(e, a) {
		t.Errorf("expected %v, but received %v", e, a)
	}
}

func TestRangeStore(t *testing.T) {
	m := newSyncMap()
	for i := 0; i < 10; i++ {
		m.Store(i, i*10)
	}

	m.Range(func(key, value interface{}) bool {
		v := value.(int)
		m.Store(key, v+1)
		return true
	})

	expectedMap := map[interface{}]interface{}{
		0: 1,
		1: 11,
		2: 21,
		3: 31,
		4: 41,
		5: 51,
		6: 61,
		7: 71,
		8: 81,
		9: 91,
	}
	actualMap := map[interface{}]interface{}{}
	m.Range(func(key, value interface{}) bool {
		actualMap[key] = value
		return true
	})

	if e, a := len(expectedMap), len(actualMap); e != a {
		t.Errorf("expected map size %d, but received %d", e, a)
	}

	if e, a := expectedMap, actualMap; !reflect.DeepEqual(e, a) {
		t.Errorf("expected %v, but received %v", e, a)
	}
}

func TestRangeGet(t *testing.T) {
	m := newSyncMap()
	for i := 0; i < 10; i++ {
		m.Store(i, i*10)
	}

	m.Range(func(key, value interface{}) bool {
		m.Load(key)
		return true
	})

	expectedMap := map[interface{}]interface{}{
		0: 0,
		1: 10,
		2: 20,
		3: 30,
		4: 40,
		5: 50,
		6: 60,
		7: 70,
		8: 80,
		9: 90,
	}
	actualMap := map[interface{}]interface{}{}
	m.Range(func(key, value interface{}) bool {
		actualMap[key] = value
		return true
	})

	if e, a := len(expectedMap), len(actualMap); e != a {
		t.Errorf("expected map size %d, but received %d", e, a)
	}

	if e, a := expectedMap, actualMap; !reflect.DeepEqual(e, a) {
		t.Errorf("expected %v, but received %v", e, a)
	}
}
