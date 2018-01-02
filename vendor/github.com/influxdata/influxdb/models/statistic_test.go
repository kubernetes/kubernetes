package models_test

import (
	"reflect"
	"testing"

	"github.com/influxdata/influxdb/models"
)

func TestTags_Merge(t *testing.T) {
	examples := []struct {
		Base   map[string]string
		Arg    map[string]string
		Result map[string]string
	}{
		{
			Base:   nil,
			Arg:    nil,
			Result: map[string]string{},
		},
		{
			Base:   nil,
			Arg:    map[string]string{"foo": "foo"},
			Result: map[string]string{"foo": "foo"},
		},
		{
			Base:   map[string]string{"foo": "foo"},
			Arg:    nil,
			Result: map[string]string{"foo": "foo"},
		},
		{
			Base:   map[string]string{"foo": "foo"},
			Arg:    map[string]string{"bar": "bar"},
			Result: map[string]string{"foo": "foo", "bar": "bar"},
		},
		{
			Base:   map[string]string{"foo": "foo", "bar": "bar"},
			Arg:    map[string]string{"zoo": "zoo"},
			Result: map[string]string{"foo": "foo", "bar": "bar", "zoo": "zoo"},
		},
		{
			Base:   map[string]string{"foo": "foo", "bar": "bar"},
			Arg:    map[string]string{"bar": "newbar"},
			Result: map[string]string{"foo": "foo", "bar": "newbar"},
		},
	}

	for i, example := range examples {
		i++
		result := models.StatisticTags(example.Base).Merge(example.Arg)
		if got, exp := result, example.Result; !reflect.DeepEqual(got, exp) {
			t.Errorf("[Example %d] got %#v, expected %#v", i, got, exp)
		}
	}
}
