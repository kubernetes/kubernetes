package types

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/coreos/go-semver/semver"
)

func TestMarshalSemver(t *testing.T) {
	tests := []struct {
		sv SemVer

		wd []byte
	}{
		{
			SemVer(semver.Version{Major: 1}),

			[]byte(`"1.0.0"`),
		},
		{
			SemVer(semver.Version{Major: 3, Minor: 2, Patch: 1}),

			[]byte(`"3.2.1"`),
		},
		{
			SemVer(semver.Version{Major: 3, Minor: 2, Patch: 1, PreRelease: "foo"}),

			[]byte(`"3.2.1-foo"`),
		},
		{
			SemVer(semver.Version{Major: 1, Minor: 2, Patch: 3, PreRelease: "alpha", Metadata: "git"}),

			[]byte(`"1.2.3-alpha+git"`),
		},
	}
	for i, tt := range tests {
		d, err := json.Marshal(tt.sv)
		if !reflect.DeepEqual(d, tt.wd) {
			t.Errorf("#%d: d=%v, want %v", i, string(d), string(tt.wd))
		}
		if err != nil {
			t.Errorf("#%d: err=%v, want nil", i, err)
		}
	}
}

func TestUnmarshalSemver(t *testing.T) {
	tests := []struct {
		d []byte

		wsv  SemVer
		werr bool
	}{
		{
			[]byte(`"1.0.0"`),

			SemVer(semver.Version{Major: 1}),
			false,
		},
		{
			[]byte(`"3.2.1"`),
			SemVer(semver.Version{Major: 3, Minor: 2, Patch: 1}),

			false,
		},
		{
			[]byte(`"3.2.1-foo"`),

			SemVer(semver.Version{Major: 3, Minor: 2, Patch: 1, PreRelease: "foo"}),
			false,
		},
		{
			[]byte(`"1.2.3-alpha+git"`),

			SemVer(semver.Version{Major: 1, Minor: 2, Patch: 3, PreRelease: "alpha", Metadata: "git"}),
			false,
		},
		{
			[]byte(`"1"`),

			SemVer{},
			true,
		},
		{
			[]byte(`"1.2.3.4"`),

			SemVer{},
			true,
		},
		{
			[]byte(`1.2.3`),

			SemVer{},
			true,
		},
		{
			[]byte(`"v1.2.3"`),

			SemVer{},
			true,
		},
	}
	for i, tt := range tests {
		var sv SemVer
		err := json.Unmarshal(tt.d, &sv)
		if !reflect.DeepEqual(sv, tt.wsv) {
			t.Errorf("#%d: semver=%#v, want %#v", i, sv, tt.wsv)
		}
		if gerr := (err != nil); gerr != tt.werr {
			t.Errorf("#%d: err==%v, want errstate %t", i, err, tt.werr)
		}
	}
}
