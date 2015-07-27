package swagger

import (
	"reflect"
	"strings"
)

func (prop *ModelProperty) setDescription(field reflect.StructField) {
	if tag := field.Tag.Get("description"); tag != "" {
		prop.Description = tag
	}
}

func (prop *ModelProperty) setDefaultValue(field reflect.StructField) {
	if tag := field.Tag.Get("default"); tag != "" {
		prop.DefaultValue = Special(tag)
	}
}

func (prop *ModelProperty) setEnumValues(field reflect.StructField) {
	// We use | to separate the enum values.  This value is chosen
	// since its unlikely to be useful in actual enumeration values.
	if tag := field.Tag.Get("enum"); tag != "" {
		prop.Enum = strings.Split(tag, "|")
	}
}

func (prop *ModelProperty) setMaximum(field reflect.StructField) {
	if tag := field.Tag.Get("maximum"); tag != "" {
		prop.Maximum = tag
	}
}

func (prop *ModelProperty) setMinimum(field reflect.StructField) {
	if tag := field.Tag.Get("minimum"); tag != "" {
		prop.Minimum = tag
	}
}

func (prop *ModelProperty) setUniqueItems(field reflect.StructField) {
	tag := field.Tag.Get("unique")
	switch tag {
	case "true":
		v := true
		prop.UniqueItems = &v
	case "false":
		v := false
		prop.UniqueItems = &v
	}
}

func (prop *ModelProperty) setPropertyMetadata(field reflect.StructField) {
	prop.setDescription(field)
	prop.setEnumValues(field)
	prop.setMinimum(field)
	prop.setMaximum(field)
	prop.setUniqueItems(field)
	prop.setDefaultValue(field)
}
