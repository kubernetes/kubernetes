package jose

import (
	"reflect"
	"testing"
	"time"
)

func TestString(t *testing.T) {
	tests := []struct {
		cl  Claims
		key string
		ok  bool
		err bool
		val string
	}{
		// ok, no err, claim exists
		{
			cl: Claims{
				"foo": "bar",
			},
			key: "foo",
			val: "bar",
			ok:  true,
			err: false,
		},
		// no claims
		{
			cl:  Claims{},
			key: "foo",
			val: "",
			ok:  false,
			err: false,
		},
		// missing claim
		{
			cl: Claims{
				"foo": "bar",
			},
			key: "xxx",
			val: "",
			ok:  false,
			err: false,
		},
		// unparsable: type
		{
			cl: Claims{
				"foo": struct{}{},
			},
			key: "foo",
			val: "",
			ok:  false,
			err: true,
		},
		// unparsable: nil value
		{
			cl: Claims{
				"foo": nil,
			},
			key: "foo",
			val: "",
			ok:  false,
			err: true,
		},
	}

	for i, tt := range tests {
		val, ok, err := tt.cl.StringClaim(tt.key)

		if tt.err && err == nil {
			t.Errorf("case %d: want err=non-nil, got err=nil", i)
		} else if !tt.err && err != nil {
			t.Errorf("case %d: want err=nil, got err=%v", i, err)
		}

		if tt.ok != ok {
			t.Errorf("case %d: want ok=%v, got ok=%v", i, tt.ok, ok)
		}

		if tt.val != val {
			t.Errorf("case %d: want val=%v, got val=%v", i, tt.val, val)
		}
	}
}

func TestInt64(t *testing.T) {
	tests := []struct {
		cl  Claims
		key string
		ok  bool
		err bool
		val int64
	}{
		// ok, no err, claim exists
		{
			cl: Claims{
				"foo": int64(100),
			},
			key: "foo",
			val: int64(100),
			ok:  true,
			err: false,
		},
		// no claims
		{
			cl:  Claims{},
			key: "foo",
			val: 0,
			ok:  false,
			err: false,
		},
		// missing claim
		{
			cl: Claims{
				"foo": "bar",
			},
			key: "xxx",
			val: 0,
			ok:  false,
			err: false,
		},
		// unparsable: type
		{
			cl: Claims{
				"foo": struct{}{},
			},
			key: "foo",
			val: 0,
			ok:  false,
			err: true,
		},
		// unparsable: nil value
		{
			cl: Claims{
				"foo": nil,
			},
			key: "foo",
			val: 0,
			ok:  false,
			err: true,
		},
	}

	for i, tt := range tests {
		val, ok, err := tt.cl.Int64Claim(tt.key)

		if tt.err && err == nil {
			t.Errorf("case %d: want err=non-nil, got err=nil", i)
		} else if !tt.err && err != nil {
			t.Errorf("case %d: want err=nil, got err=%v", i, err)
		}

		if tt.ok != ok {
			t.Errorf("case %d: want ok=%v, got ok=%v", i, tt.ok, ok)
		}

		if tt.val != val {
			t.Errorf("case %d: want val=%v, got val=%v", i, tt.val, val)
		}
	}
}

func TestTime(t *testing.T) {
	now := time.Now().UTC()
	unixNow := now.Unix()

	tests := []struct {
		cl  Claims
		key string
		ok  bool
		err bool
		val time.Time
	}{
		// ok, no err, claim exists
		{
			cl: Claims{
				"foo": unixNow,
			},
			key: "foo",
			val: time.Unix(now.Unix(), 0).UTC(),
			ok:  true,
			err: false,
		},
		// no claims
		{
			cl:  Claims{},
			key: "foo",
			val: time.Time{},
			ok:  false,
			err: false,
		},
		// missing claim
		{
			cl: Claims{
				"foo": "bar",
			},
			key: "xxx",
			val: time.Time{},
			ok:  false,
			err: false,
		},
		// unparsable: type
		{
			cl: Claims{
				"foo": struct{}{},
			},
			key: "foo",
			val: time.Time{},
			ok:  false,
			err: true,
		},
		// unparsable: nil value
		{
			cl: Claims{
				"foo": nil,
			},
			key: "foo",
			val: time.Time{},
			ok:  false,
			err: true,
		},
	}

	for i, tt := range tests {
		val, ok, err := tt.cl.TimeClaim(tt.key)

		if tt.err && err == nil {
			t.Errorf("case %d: want err=non-nil, got err=nil", i)
		} else if !tt.err && err != nil {
			t.Errorf("case %d: want err=nil, got err=%v", i, err)
		}

		if tt.ok != ok {
			t.Errorf("case %d: want ok=%v, got ok=%v", i, tt.ok, ok)
		}

		if tt.val != val {
			t.Errorf("case %d: want val=%v, got val=%v", i, tt.val, val)
		}
	}
}

func TestStringArray(t *testing.T) {
	tests := []struct {
		cl  Claims
		key string
		ok  bool
		err bool
		val []string
	}{
		// ok, no err, claim exists
		{
			cl: Claims{
				"foo": []string{"bar", "faf"},
			},
			key: "foo",
			val: []string{"bar", "faf"},
			ok:  true,
			err: false,
		},
		// ok, no err, []interface{}
		{
			cl: Claims{
				"foo": []interface{}{"bar", "faf"},
			},
			key: "foo",
			val: []string{"bar", "faf"},
			ok:  true,
			err: false,
		},
		// no claims
		{
			cl:  Claims{},
			key: "foo",
			val: nil,
			ok:  false,
			err: false,
		},
		// missing claim
		{
			cl: Claims{
				"foo": "bar",
			},
			key: "xxx",
			val: nil,
			ok:  false,
			err: false,
		},
		// unparsable: type
		{
			cl: Claims{
				"foo": struct{}{},
			},
			key: "foo",
			val: nil,
			ok:  false,
			err: true,
		},
		// unparsable: nil value
		{
			cl: Claims{
				"foo": nil,
			},
			key: "foo",
			val: nil,
			ok:  false,
			err: true,
		},
	}

	for i, tt := range tests {
		val, ok, err := tt.cl.StringsClaim(tt.key)

		if tt.err && err == nil {
			t.Errorf("case %d: want err=non-nil, got err=nil", i)
		} else if !tt.err && err != nil {
			t.Errorf("case %d: want err=nil, got err=%v", i, err)
		}

		if tt.ok != ok {
			t.Errorf("case %d: want ok=%v, got ok=%v", i, tt.ok, ok)
		}

		if !reflect.DeepEqual(tt.val, val) {
			t.Errorf("case %d: want val=%v, got val=%v", i, tt.val, val)
		}
	}
}
