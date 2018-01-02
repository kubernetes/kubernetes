// Copyright 2015 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package search

import (
	"reflect"
	"testing"
)

func TestLoadingStruct(t *testing.T) {
	testCases := []struct {
		desc    string
		fields  []Field
		meta    *DocumentMetadata
		want    interface{}
		wantErr bool
	}{
		{
			desc: "Basic struct",
			fields: []Field{
				{Name: "Name", Value: "Gopher"},
				{Name: "Legs", Value: float64(4)},
			},
			want: &struct {
				Name string
				Legs float64
			}{"Gopher", 4},
		},
		{
			desc: "Struct with tags",
			fields: []Field{
				{Name: "Name", Value: "Gopher"},
				{Name: "about", Value: "Likes slide rules."},
			},
			meta: &DocumentMetadata{Facets: []Facet{
				{Name: "Legs", Value: float64(4)},
				{Name: "Fur", Value: Atom("furry")},
			}},
			want: &struct {
				Name string
				Info string  `search:"about"`
				Legs float64 `search:",facet"`
				Fuzz Atom    `search:"Fur,facet"`
			}{"Gopher", "Likes slide rules.", 4, Atom("furry")},
		},
		{
			desc: "Bad field from tag",
			want: &struct {
				AlphaBeta string `search:"αβ"`
			}{},
			wantErr: true,
		},
		{
			desc: "Ignore missing field",
			fields: []Field{
				{Name: "Meaning", Value: float64(42)},
			},
			want:    &struct{}{},
			wantErr: true,
		},
		{
			desc: "Ignore unsettable field",
			fields: []Field{
				{Name: "meaning", Value: float64(42)},
			},
			want:    &struct{ meaning float64 }{}, // field not populated.
			wantErr: true,
		},
		{
			desc: "Error on missing facet",
			meta: &DocumentMetadata{Facets: []Facet{
				{Name: "Set", Value: Atom("yes")},
				{Name: "Missing", Value: Atom("no")},
			}},
			want: &struct {
				Set Atom `search:",facet"`
			}{Atom("yes")},
			wantErr: true,
		},
		{
			desc: "Error on unsettable facet",
			meta: &DocumentMetadata{Facets: []Facet{
				{Name: "Set", Value: Atom("yes")},
				{Name: "unset", Value: Atom("no")},
			}},
			want: &struct {
				Set Atom `search:",facet"`
			}{Atom("yes")},
			wantErr: true,
		},
		{
			desc: "Error setting ignored field",
			fields: []Field{
				{Name: "Set", Value: "yes"},
				{Name: "Ignored", Value: "no"},
			},
			want: &struct {
				Set     string
				Ignored string `search:"-"`
			}{Set: "yes"},
			wantErr: true,
		},
		{
			desc: "Error setting ignored facet",
			meta: &DocumentMetadata{Facets: []Facet{
				{Name: "Set", Value: Atom("yes")},
				{Name: "Ignored", Value: Atom("no")},
			}},
			want: &struct {
				Set     Atom `search:",facet"`
				Ignored Atom `search:"-,facet"`
			}{Set: Atom("yes")},
			wantErr: true,
		},
	}

	for _, tt := range testCases {
		// Make a pointer to an empty version of what want points to.
		dst := reflect.New(reflect.TypeOf(tt.want).Elem()).Interface()
		err := loadStructWithMeta(dst, tt.fields, tt.meta)
		if err != nil != tt.wantErr {
			t.Errorf("%s: got err %v; want err %t", tt.desc, err, tt.wantErr)
			continue
		}
		if !reflect.DeepEqual(dst, tt.want) {
			t.Errorf("%s: doesn't match\ngot:  %v\nwant: %v", tt.desc, dst, tt.want)
		}
	}
}

func TestSavingStruct(t *testing.T) {
	testCases := []struct {
		desc       string
		doc        interface{}
		wantFields []Field
		wantFacets []Facet
	}{
		{
			desc: "Basic struct",
			doc: &struct {
				Name string
				Legs float64
			}{"Gopher", 4},
			wantFields: []Field{
				{Name: "Name", Value: "Gopher"},
				{Name: "Legs", Value: float64(4)},
			},
		},
		{
			desc: "Struct with tags",
			doc: &struct {
				Name string
				Info string  `search:"about"`
				Legs float64 `search:",facet"`
				Fuzz Atom    `search:"Fur,facet"`
			}{"Gopher", "Likes slide rules.", 4, Atom("furry")},
			wantFields: []Field{
				{Name: "Name", Value: "Gopher"},
				{Name: "about", Value: "Likes slide rules."},
			},
			wantFacets: []Facet{
				{Name: "Legs", Value: float64(4)},
				{Name: "Fur", Value: Atom("furry")},
			},
		},
		{
			desc: "Ignore unexported struct fields",
			doc: &struct {
				Name string
				info string
				Legs float64 `search:",facet"`
				fuzz Atom    `search:",facet"`
			}{"Gopher", "Likes slide rules.", 4, Atom("furry")},
			wantFields: []Field{
				{Name: "Name", Value: "Gopher"},
			},
			wantFacets: []Facet{
				{Name: "Legs", Value: float64(4)},
			},
		},
		{
			desc: "Ignore fields marked -",
			doc: &struct {
				Name string
				Info string  `search:"-"`
				Legs float64 `search:",facet"`
				Fuzz Atom    `search:"-,facet"`
			}{"Gopher", "Likes slide rules.", 4, Atom("furry")},
			wantFields: []Field{
				{Name: "Name", Value: "Gopher"},
			},
			wantFacets: []Facet{
				{Name: "Legs", Value: float64(4)},
			},
		},
	}

	for _, tt := range testCases {
		fields, meta, err := saveStructWithMeta(tt.doc)
		if err != nil {
			t.Errorf("%s: got err %v; want nil", tt.desc, err)
			continue
		}
		if !reflect.DeepEqual(fields, tt.wantFields) {
			t.Errorf("%s: fields don't match\ngot:  %v\nwant: %v", tt.desc, fields, tt.wantFields)
		}
		if facets := meta.Facets; !reflect.DeepEqual(facets, tt.wantFacets) {
			t.Errorf("%s: facets don't match\ngot:  %v\nwant: %v", tt.desc, facets, tt.wantFacets)
		}
	}
}
