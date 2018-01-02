package jsonpath

import (
	"bytes"
	"fmt"
	"io"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestPathActionSingleMatch(t *testing.T) {

	j := []byte(`
	{
		"foo": 1,
		"bar": 2,
		"test": "Hello, world!",
		"baz": 123.1,
		"array": [
			{"foo": 1},
			{"bar": 2},
			{"baz": 3}
		],
		"subobj": {
			"foo": 1,
			"subarray": [1,2,3],
			"subsubobj": {
				"bar": 2,
				"baz": 3,
				"array": ["hello", "world"]
			}
		},
		"bool": true
	}`)

	decodeCount := 0
	decode := func(d *Decoder) interface{} {
		decodeCount++
		var v interface{}
		err := d.Decode(&v)
		assert.NoError(t, err)
		return v
	}

	dc := NewDecoder(bytes.NewBuffer(j))
	actions := &PathActions{}

	actions.Add(func(d *Decoder) error {
		assert.Equal(t, float64(2), decode(d))
		return nil
	}, "array", 1, "bar")

	actions.Add(func(d *Decoder) error {
		assert.Equal(t, "Hello, world!", decode(d))
		return nil
	}, "test")

	actions.Add(func(d *Decoder) error {
		assert.Equal(t, []interface{}{float64(1), float64(2), float64(3)}, decode(d))
		return nil
	}, "subobj", "subarray")

	actions.Add(func(d *Decoder) error {
		assert.Equal(t, float64(1), decode(d))
		return nil
	}, "foo")

	actions.Add(func(d *Decoder) error {
		assert.Equal(t, float64(2), decode(d))
		return nil
	}, "bar")

	dc.Scan(actions)

	assert.Equal(t, 5, decodeCount)
}

func TestPathActionAnyIndex(t *testing.T) {

	j := []byte(`
	{
		"foo": 1,
		"bar": 2,
		"test": "Hello, world!",
		"baz": 123.1,
		"array": [
			{"num": 1},
			{"num": 2},
			{"num": 3}
		],
		"subobj": {
			"foo": 1,
			"subarray": [1,2,3],
			"subsubobj": {
				"bar": 2,
				"baz": 3,
				"array": ["hello", "world"]
			}
		},
		"bool": true
	}`)

	dc := NewDecoder(bytes.NewBuffer(j))
	actions := &PathActions{}

	numbers := []int{}
	actions.Add(func(d *Decoder) (err error) {
		var v int
		err = d.Decode(&v)
		require.NoError(t, err)
		numbers = append(numbers, v)
		return
	}, "array", AnyIndex, "num")

	numbers2 := []int{}
	actions.Add(func(d *Decoder) (err error) {
		var v int
		err = d.Decode(&v)
		require.NoError(t, err)
		numbers2 = append(numbers2, v)
		return
	}, "subobj", "subarray", AnyIndex)

	strings := []string{}
	actions.Add(func(d *Decoder) (err error) {
		var v string
		err = d.Decode(&v)
		require.NoError(t, err)
		strings = append(strings, v)
		return
	}, "subobj", "subsubobj", "array", AnyIndex)

	dc.Scan(actions)

	assert.Equal(t, []int{1, 2, 3}, numbers)
	assert.Equal(t, []int{1, 2, 3}, numbers2)
	assert.Equal(t, []string{"hello", "world"}, strings)
}

func TestPathActionJsonStream(t *testing.T) {

	j := []byte(`
	{
    "make": "Porsche",
		"model": "356 Coupé",
    "years": { "from": 1948, "to": 1965}
  }
  {
    "years": { "from": 1964, "to": 1969},
    "make": "Ford",
    "model": "GT40"
  }
  {
    "make": "Ferrari",
    "model": "308 GTB",
    "years": { "to": 1985, "from": 1975}
  }
  `)

	dc := NewDecoder(bytes.NewBuffer(j))

	var from, to []int
	actions := &PathActions{}
	actions.Add(func(d *Decoder) (err error) {
		var v int
		err = d.Decode(&v)
		require.NoError(t, err)
		from = append(from, v)
		return
	}, "years", "from")
	actions.Add(func(d *Decoder) (err error) {
		var v int
		err = d.Decode(&v)
		require.NoError(t, err)
		to = append(to, v)
		return
	}, "years", "to")

	var err error
	var ok = true
	for ok && err == nil {
		ok, err = dc.Scan(actions)
		if err != io.EOF {
			require.NoError(t, err)
		}
	}

	assert.Equal(t, []int{1948, 1964, 1975}, from)
	assert.Equal(t, []int{1965, 1969, 1985}, to)
}

func TestPathActionJsonSubObjects(t *testing.T) {

	j := []byte(`
    {
      "set": "cars",
    	"data": [
        {
          "make": "Porsche",
      		"model": "356 Coupé",
          "years": { "from": 1948, "to": 1965}
        },
        {
          "years": { "from": 1964, "to": 1969},
          "make": "Ford",
          "model": "GT40"
        },
        {
          "make": "Ferrari",
          "model": "308 GTB",
          "years": { "to": 1985, "from": 1975}
        }
      ],
      "more": true
    }
  `)

	dc := NewDecoder(bytes.NewBuffer(j))

	var from, to []int
	actions := &PathActions{}
	actions.Add(func(d *Decoder) (err error) {
		var v int
		err = d.Decode(&v)
		require.NoError(t, err)
		from = append(from, v)
		return
	}, "data", AnyIndex, "years", "from")
	actions.Add(func(d *Decoder) (err error) {
		var v int
		err = d.Decode(&v)
		require.NoError(t, err)
		to = append(to, v)
		return
	}, "data", AnyIndex, "years", "to")

	var err error
	var ok = true
	for ok && err == nil {
		_, err = dc.Scan(actions)
		if err != io.EOF {
			require.NoError(t, err)
		}
	}

	assert.Equal(t, []int{1948, 1964, 1975}, from)
	assert.Equal(t, []int{1965, 1969, 1985}, to)
}

func TestPathActionSeekThenScan(t *testing.T) {

	j := []byte(`
    {
      "set": "cars",
    	"data": [
        {
          "make": "Porsche",
      		"model": "356 Coupé",
          "years": { "from": 1948, "to": 1965}
        },
        {
          "years": { "from": 1964, "to": 1969},
          "make": "Ford",
          "model": "GT40"
        },
        {
          "make": "Ferrari",
          "model": "308 GTB",
          "years": { "to": 1985, "from": 1975}
        }
      ],
      "more": true
    }
  `)

	dc := NewDecoder(bytes.NewBuffer(j))
	ok, err := dc.SeekTo("data", 0)
	require.NoError(t, err)
	require.True(t, ok)

	var from, to int
	actions := &PathActions{}
	actions.Add(func(d *Decoder) (err error) {
		err = d.Decode(&from)
		require.NoError(t, err)
		return
	}, "years", "from")
	actions.Add(func(d *Decoder) (err error) {
		err = d.Decode(&to)
		require.NoError(t, err)
		return
	}, "years", "to")

	outs := []string{}
	for ok && err == nil {
		ok, err = dc.Scan(actions)
		if err != io.EOF {
			require.NoError(t, err)
		}
		if err == nil || err == io.EOF {
			outs = append(outs, fmt.Sprintf("%v-%v", from, to))
		}
	}

	assert.Equal(t, []string{"1948-1965", "1964-1969", "1975-1985"}, outs)
}

func TestPathActionSeekOffsetThenScan(t *testing.T) {

	j := []byte(`
    {
      "set": "cars",
    	"data": [
        {
          "make": "Porsche",
      		"model": "356 Coupé",
          "years": { "from": 1948, "to": 1965}
        },
        {
          "years": { "from": 1964, "to": 1969},
          "make": "Ford",
          "model": "GT40"
        },
        {
          "make": "Ferrari",
          "model": "308 GTB",
          "years": { "to": 1985, "from": 1975}
        }
      ],
      "more": true
    }
  `)

	dc := NewDecoder(bytes.NewBuffer(j))
	ok, err := dc.SeekTo("data", 1)
	require.NoError(t, err)
	require.True(t, ok)

	var from, to int
	actions := &PathActions{}
	actions.Add(func(d *Decoder) (err error) {
		err = d.Decode(&from)
		require.NoError(t, err)
		return
	}, "years", "from")
	actions.Add(func(d *Decoder) (err error) {
		err = d.Decode(&to)
		require.NoError(t, err)
		return
	}, "years", "to")

	outs := []string{}
	for ok && err == nil {
		ok, err = dc.Scan(actions)
		if err != io.EOF {
			require.NoError(t, err)
		}
		if err == nil || err == io.EOF {
			outs = append(outs, fmt.Sprintf("%v-%v", from, to))
		}
	}

	assert.Equal(t, []string{"1964-1969", "1975-1985"}, outs)
}

func TestPathActionSeekThenScanThenScan(t *testing.T) {

	j := []byte(`
    {
      "set": "cars",
    	"data": [
        {
          "make": "Porsche",
      		"model": "356 Coupé",
          "years": { "from": 1948, "to": 1965}
        },
        {
          "years": { "from": 1964, "to": 1969},
          "make": "Ford",
          "model": "GT40"
        }
      ],
      "more": [
        {
          "make": "Ferrari",
          "model": "308 GTB",
          "years": { "to": 1985, "from": 1975}
        }
      ]
    }
  `)

	dc := NewDecoder(bytes.NewBuffer(j))
	ok, err := dc.SeekTo("data", 0)
	require.NoError(t, err)
	require.True(t, ok)

	var from, to int
	actions := &PathActions{}
	actions.Add(func(d *Decoder) (err error) {
		err = d.Decode(&from)
		require.NoError(t, err)
		return
	}, "years", "from")
	actions.Add(func(d *Decoder) (err error) {
		err = d.Decode(&to)
		require.NoError(t, err)
		return
	}, "years", "to")

	outs := []string{}
	for ok && err == nil {
		ok, err = dc.Scan(actions)
		if err != io.EOF {
			require.NoError(t, err)
		}
		if err == nil || err == io.EOF {
			outs = append(outs, fmt.Sprintf("%v-%v", from, to))
		}
	}

	assert.Equal(t, []string{"1948-1965", "1964-1969"}, outs)

	ok, err = dc.SeekTo("more", 0)
	require.NoError(t, err)
	require.True(t, ok)
	outs = []string{}
	for ok && err == nil {
		ok, err = dc.Scan(actions)
		if err != io.EOF {
			require.NoError(t, err)
		}
		if err == nil || err == io.EOF {
			outs = append(outs, fmt.Sprintf("%v-%v", from, to))
		}
	}

	assert.Equal(t, []string{"1975-1985"}, outs)
}

func TestPathActionSeekThenScanHetero(t *testing.T) {

	j := []byte(`
    {
      "set": "cars",
    	"data": [
        {
          "make": "Porsche",
      		"model": "356 Coupé",
          "years": { "from": 1948, "to": 1965}
        },
        ["other","random","stuff"],
        {
          "years": { "from": 1964, "to": 1969},
          "make": "Ford",
          "model": "GT40"
        },
        {},
        "str",
        {
          "make": "Ferrari",
          "model": "308 GTB",
          "years": { "to": 1985, "from": 1975}
        }
      ],
      "more": true
    }
  `)

	dc := NewDecoder(bytes.NewBuffer(j))
	ok, err := dc.SeekTo("data", 0)
	require.NoError(t, err)
	require.True(t, ok)

	var from, to int
	actions := &PathActions{}
	actions.Add(func(d *Decoder) (err error) {
		err = d.Decode(&from)
		require.NoError(t, err)
		return
	}, "years", "from")
	actions.Add(func(d *Decoder) (err error) {
		err = d.Decode(&to)
		require.NoError(t, err)
		return
	}, "years", "to")

	outs := []string{}
	for ok && err == nil {
		ok, err = dc.Scan(actions)
		if err != io.EOF {
			require.NoError(t, err)
		}
		if (err == nil || err == io.EOF) && (from != 0 && to != 0) {
			outs = append(outs, fmt.Sprintf("%v-%v", from, to))
			from, to = 0, 0
		}
	}

	assert.Equal(t, []string{"1948-1965", "1964-1969", "1975-1985"}, outs)
}

func TestPathActionNested(t *testing.T) {

	j := []byte(`
    {
      "set": "cars",
    	"data": [
        {
          "make": "Porsche",
      		"model": "356 Coupé",
          "years": { "from": 1948, "to": 1965}
        },
        {
          "years": { "from": 1964, "to": 1969},
          "make": "Ford",
          "model": "GT40"
        },
        {
          "make": "Ferrari",
          "model": "308 GTB",
          "years": { "to": 1985, "from": 1975}
        }
      ],
      "more": true
    }
  `)

	var from, to int
	caractions := &PathActions{}
	caractions.Add(func(d *Decoder) (err error) {
		err = d.Decode(&from)
		require.NoError(t, err)
		return
	}, "years", "from")
	caractions.Add(func(d *Decoder) (err error) {
		err = d.Decode(&to)
		require.NoError(t, err)
		return
	}, "years", "to")

	outs := []string{}

	actions := &PathActions{}
	actions.Add(func(d *Decoder) error {

		_, err := d.Scan(caractions)
		if err != nil {
			return err
		}
		outs = append(outs, fmt.Sprintf("%v-%v", from, to))
		return nil

	}, "data", AnyIndex)

	ok, err := NewDecoder(bytes.NewBuffer(j)).Scan(actions)
	assert.NoError(t, err)
	assert.False(t, ok)

	assert.Equal(t, []string{"1948-1965", "1964-1969", "1975-1985"}, outs)
}
