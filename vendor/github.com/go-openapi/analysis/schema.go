package analysis

import (
	"github.com/go-openapi/spec"
	"github.com/go-openapi/strfmt"
)

// SchemaOpts configures the schema analyzer
type SchemaOpts struct {
	Schema   *spec.Schema
	Root     interface{}
	BasePath string
	_        struct{}
}

// Schema analysis, will classify the schema according to known
// patterns.
func Schema(opts SchemaOpts) (*AnalyzedSchema, error) {
	a := &AnalyzedSchema{
		schema:   opts.Schema,
		root:     opts.Root,
		basePath: opts.BasePath,
	}

	a.initializeFlags()
	a.inferKnownType()
	a.inferEnum()
	a.inferBaseType()

	if err := a.inferMap(); err != nil {
		return nil, err
	}
	if err := a.inferArray(); err != nil {
		return nil, err
	}

	if err := a.inferTuple(); err != nil {
		return nil, err
	}

	if err := a.inferFromRef(); err != nil {
		return nil, err
	}

	a.inferSimpleSchema()
	return a, nil
}

// AnalyzedSchema indicates what the schema represents
type AnalyzedSchema struct {
	schema   *spec.Schema
	root     interface{}
	basePath string

	hasProps           bool
	hasAllOf           bool
	hasItems           bool
	hasAdditionalProps bool
	hasAdditionalItems bool
	hasRef             bool

	IsKnownType      bool
	IsSimpleSchema   bool
	IsArray          bool
	IsSimpleArray    bool
	IsMap            bool
	IsSimpleMap      bool
	IsExtendedObject bool
	IsTuple          bool
	IsTupleWithExtra bool
	IsBaseType       bool
	IsEnum           bool
}

// Inherits copies value fields from other onto this schema
func (a *AnalyzedSchema) inherits(other *AnalyzedSchema) {
	if other == nil {
		return
	}
	a.hasProps = other.hasProps
	a.hasAllOf = other.hasAllOf
	a.hasItems = other.hasItems
	a.hasAdditionalItems = other.hasAdditionalItems
	a.hasAdditionalProps = other.hasAdditionalProps
	a.hasRef = other.hasRef

	a.IsKnownType = other.IsKnownType
	a.IsSimpleSchema = other.IsSimpleSchema
	a.IsArray = other.IsArray
	a.IsSimpleArray = other.IsSimpleArray
	a.IsMap = other.IsMap
	a.IsSimpleMap = other.IsSimpleMap
	a.IsExtendedObject = other.IsExtendedObject
	a.IsTuple = other.IsTuple
	a.IsTupleWithExtra = other.IsTupleWithExtra
	a.IsBaseType = other.IsBaseType
	a.IsEnum = other.IsEnum
}

func (a *AnalyzedSchema) inferFromRef() error {
	if a.hasRef {
		sch := new(spec.Schema)
		sch.Ref = a.schema.Ref
		err := spec.ExpandSchema(sch, a.root, nil)
		if err != nil {
			return err
		}
		if sch != nil {
			rsch, err := Schema(SchemaOpts{
				Schema:   sch,
				Root:     a.root,
				BasePath: a.basePath,
			})
			if err != nil {
				return err
			}
			a.inherits(rsch)
		}
	}
	return nil
}

func (a *AnalyzedSchema) inferSimpleSchema() {
	a.IsSimpleSchema = a.IsKnownType || a.IsSimpleArray || a.IsSimpleMap
}

func (a *AnalyzedSchema) inferKnownType() {
	tpe := a.schema.Type
	format := a.schema.Format
	a.IsKnownType = tpe.Contains("boolean") ||
		tpe.Contains("integer") ||
		tpe.Contains("number") ||
		tpe.Contains("string") ||
		(format != "" && strfmt.Default.ContainsName(format)) ||
		(a.isObjectType() && !a.hasProps && !a.hasAllOf && !a.hasAdditionalProps && !a.hasAdditionalItems)
}

func (a *AnalyzedSchema) inferMap() error {
	if a.isObjectType() {
		hasExtra := a.hasProps || a.hasAllOf
		a.IsMap = a.hasAdditionalProps && !hasExtra
		a.IsExtendedObject = a.hasAdditionalProps && hasExtra
		if a.IsMap {
			if a.schema.AdditionalProperties.Schema != nil {
				msch, err := Schema(SchemaOpts{
					Schema:   a.schema.AdditionalProperties.Schema,
					Root:     a.root,
					BasePath: a.basePath,
				})
				if err != nil {
					return err
				}
				a.IsSimpleMap = msch.IsSimpleSchema
			} else if a.schema.AdditionalProperties.Allows {
				a.IsSimpleMap = true
			}
		}
	}
	return nil
}

func (a *AnalyzedSchema) inferArray() error {
	fromValid := a.isArrayType() && (a.schema.Items == nil || a.schema.Items.Len() < 2)
	a.IsArray = fromValid || (a.hasItems && a.schema.Items.Len() < 2)
	if a.IsArray && a.hasItems {
		if a.schema.Items.Schema != nil {
			itsch, err := Schema(SchemaOpts{
				Schema:   a.schema.Items.Schema,
				Root:     a.root,
				BasePath: a.basePath,
			})
			if err != nil {
				return err
			}
			a.IsSimpleArray = itsch.IsSimpleSchema
		}
		if len(a.schema.Items.Schemas) > 0 {
			itsch, err := Schema(SchemaOpts{
				Schema:   &a.schema.Items.Schemas[0],
				Root:     a.root,
				BasePath: a.basePath,
			})
			if err != nil {
				return err
			}
			a.IsSimpleArray = itsch.IsSimpleSchema
		}
	}
	if a.IsArray && !a.hasItems {
		a.IsSimpleArray = true
	}
	return nil
}

func (a *AnalyzedSchema) inferTuple() error {
	tuple := a.hasItems && a.schema.Items.Len() > 1
	a.IsTuple = tuple && !a.hasAdditionalItems
	a.IsTupleWithExtra = tuple && a.hasAdditionalItems
	return nil
}

func (a *AnalyzedSchema) inferBaseType() {
	if a.isObjectType() {
		a.IsBaseType = a.schema.Discriminator != ""
	}
}

func (a *AnalyzedSchema) inferEnum() {
	a.IsEnum = len(a.schema.Enum) > 0
}

func (a *AnalyzedSchema) initializeFlags() {
	a.hasProps = len(a.schema.Properties) > 0
	a.hasAllOf = len(a.schema.AllOf) > 0
	a.hasRef = a.schema.Ref.String() != ""

	a.hasItems = a.schema.Items != nil &&
		(a.schema.Items.Schema != nil || len(a.schema.Items.Schemas) > 0)

	a.hasAdditionalProps = a.schema.AdditionalProperties != nil &&
		(a.schema.AdditionalProperties != nil || a.schema.AdditionalProperties.Allows)

	a.hasAdditionalItems = a.schema.AdditionalItems != nil &&
		(a.schema.AdditionalItems.Schema != nil || a.schema.AdditionalItems.Allows)

}

func (a *AnalyzedSchema) isObjectType() bool {
	return !a.hasRef && (a.schema.Type == nil || a.schema.Type.Contains("") || a.schema.Type.Contains("object"))
}

func (a *AnalyzedSchema) isArrayType() bool {
	return !a.hasRef && (a.schema.Type != nil && a.schema.Type.Contains("array"))
}
