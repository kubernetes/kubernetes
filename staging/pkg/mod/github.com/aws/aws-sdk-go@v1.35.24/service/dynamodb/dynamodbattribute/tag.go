package dynamodbattribute

import (
	"reflect"
	"strings"
)

type tag struct {
	Name                         string
	Ignore                       bool
	OmitEmpty                    bool
	OmitEmptyElem                bool
	AsString                     bool
	AsBinSet, AsNumSet, AsStrSet bool
	AsUnixTime                   bool
}

func (t *tag) parseAVTag(structTag reflect.StructTag) {
	tagStr := structTag.Get("dynamodbav")
	if len(tagStr) == 0 {
		return
	}

	t.parseTagStr(tagStr)
}

func (t *tag) parseStructTag(tag string, structTag reflect.StructTag) {
	tagStr := structTag.Get(tag)
	if len(tagStr) == 0 {
		return
	}

	t.parseTagStr(tagStr)
}

func (t *tag) parseTagStr(tagStr string) {
	parts := strings.Split(tagStr, ",")
	if len(parts) == 0 {
		return
	}

	if name := parts[0]; name == "-" {
		t.Name = ""
		t.Ignore = true
	} else {
		t.Name = name
		t.Ignore = false
	}

	for _, opt := range parts[1:] {
		switch opt {
		case "omitempty":
			t.OmitEmpty = true
		case "omitemptyelem":
			t.OmitEmptyElem = true
		case "string":
			t.AsString = true
		case "binaryset":
			t.AsBinSet = true
		case "numberset":
			t.AsNumSet = true
		case "stringset":
			t.AsStrSet = true
		case "unixtime":
			t.AsUnixTime = true
		}
	}
}
