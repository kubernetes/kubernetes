// +build codegen

package api

import (
	"fmt"
	"regexp"
	"strings"
)

// updateTopLevelShapeReferences moves resultWrapper, locationName, and
// xmlNamespace traits from toplevel shape references to the toplevel
// shapes for easier code generation
func (a *API) updateTopLevelShapeReferences() {
	for _, o := range a.Operations {
		// these are for REST-XML services
		if o.InputRef.LocationName != "" {
			o.InputRef.Shape.LocationName = o.InputRef.LocationName
		}
		if o.InputRef.Location != "" {
			o.InputRef.Shape.Location = o.InputRef.Location
		}
		if o.InputRef.Payload != "" {
			o.InputRef.Shape.Payload = o.InputRef.Payload
		}
		if o.InputRef.XMLNamespace.Prefix != "" {
			o.InputRef.Shape.XMLNamespace.Prefix = o.InputRef.XMLNamespace.Prefix
		}
		if o.InputRef.XMLNamespace.URI != "" {
			o.InputRef.Shape.XMLNamespace.URI = o.InputRef.XMLNamespace.URI
		}
	}

}

// writeShapeNames sets each shape's API and shape name values. Binding the
// shape to its parent API.
func (a *API) writeShapeNames() {
	for n, s := range a.Shapes {
		s.API = a
		s.ShapeName = n
	}
}

func (a *API) resolveReferences() {
	resolver := referenceResolver{API: a, visited: map[*ShapeRef]bool{}}

	for _, s := range a.Shapes {
		resolver.resolveShape(s)
	}

	for _, o := range a.Operations {
		o.API = a // resolve parent reference

		resolver.resolveReference(&o.InputRef)
		resolver.resolveReference(&o.OutputRef)

		// Resolve references for errors also
		for i := range o.ErrorRefs {
			resolver.resolveReference(&o.ErrorRefs[i])
			o.ErrorRefs[i].Shape.IsError = true
		}
	}
}

// A referenceResolver provides a way to resolve shape references to
// shape definitions.
type referenceResolver struct {
	*API
	visited map[*ShapeRef]bool
}

var jsonvalueShape = &Shape{
	ShapeName: "JSONValue",
	Type:      "jsonvalue",
	ValueRef: ShapeRef{
		JSONValue: true,
	},
}

// resolveReference updates a shape reference to reference the API and
// its shape definition. All other nested references are also resolved.
func (r *referenceResolver) resolveReference(ref *ShapeRef) {
	if ref.ShapeName == "" {
		return
	}

	if shape, ok := r.API.Shapes[ref.ShapeName]; ok {
		if ref.JSONValue {
			ref.ShapeName = "JSONValue"
			r.API.Shapes[ref.ShapeName] = jsonvalueShape
		}

		ref.API = r.API   // resolve reference back to API
		ref.Shape = shape // resolve shape reference

		if r.visited[ref] {
			return
		}
		r.visited[ref] = true

		shape.refs = append(shape.refs, ref) // register the ref

		// resolve shape's references, if it has any
		r.resolveShape(shape)
	}
}

// resolveShape resolves a shape's Member Key Value, and nested member
// shape references.
func (r *referenceResolver) resolveShape(shape *Shape) {
	r.resolveReference(&shape.MemberRef)
	r.resolveReference(&shape.KeyRef)
	r.resolveReference(&shape.ValueRef)
	for _, m := range shape.MemberRefs {
		r.resolveReference(m)
	}
}

// renameToplevelShapes renames all top level shapes of an API to their
// exportable variant. The shapes are also updated to include notations
// if they are Input or Outputs.
func (a *API) renameToplevelShapes() {
	for _, v := range a.OperationList() {
		if v.HasInput() {
			name := v.ExportedName + "Input"
			switch {
			case a.Shapes[name] == nil:
				if service, ok := shamelist[a.name]; ok {
					if check, ok := service[v.Name]; ok && check.input {
						break
					}
				}
				v.InputRef.Shape.Rename(name)
			}
		}
		if v.HasOutput() {
			name := v.ExportedName + "Output"
			switch {
			case a.Shapes[name] == nil:
				if service, ok := shamelist[a.name]; ok {
					if check, ok := service[v.Name]; ok && check.output {
						break
					}
				}
				v.OutputRef.Shape.Rename(name)
			}
		}
		v.InputRef.Payload = a.ExportableName(v.InputRef.Payload)
		v.OutputRef.Payload = a.ExportableName(v.OutputRef.Payload)
	}
}

// fixStutterNames fixes all name struttering based on Go naming conventions.
// "Stuttering" is when the prefix of a structure or function matches the
// package name (case insensitive).
func (a *API) fixStutterNames() {
	str, end := a.StructName(), ""
	if len(str) > 1 {
		l := len(str) - 1
		str, end = str[0:l], str[l:]
	}
	re := regexp.MustCompile(fmt.Sprintf(`\A(?i:%s)%s`, str, end))

	for name, op := range a.Operations {
		newName := re.ReplaceAllString(name, "")
		if newName != name {
			delete(a.Operations, name)
			a.Operations[newName] = op
		}
		op.ExportedName = newName
	}

	for k, s := range a.Shapes {
		newName := re.ReplaceAllString(k, "")
		if newName != s.ShapeName {
			s.Rename(newName)
		}
	}
}

// renameExportable renames all operation names to be exportable names.
// All nested Shape names are also updated to the exportable variant.
func (a *API) renameExportable() {
	for name, op := range a.Operations {
		newName := a.ExportableName(name)
		if newName != name {
			delete(a.Operations, name)
			a.Operations[newName] = op
		}
		op.ExportedName = newName
	}

	for k, s := range a.Shapes {
		// FIXME SNS has lower and uppercased shape names with the same name,
		// except the lowercased variant is used exclusively for string and
		// other primitive types. Renaming both would cause a collision.
		// We work around this by only renaming the structure shapes.
		if s.Type == "string" {
			continue
		}

		for mName, member := range s.MemberRefs {
			newName := a.ExportableName(mName)
			if newName != mName {
				delete(s.MemberRefs, mName)
				s.MemberRefs[newName] = member

				// also apply locationName trait so we keep the old one
				// but only if there's no locationName trait on ref or shape
				if member.LocationName == "" && member.Shape.LocationName == "" {
					member.LocationName = mName
				}
			}

			if newName == "_" {
				panic("Shape " + s.ShapeName + " uses reserved member name '_'")
			}
		}

		newName := a.ExportableName(k)
		if newName != s.ShapeName {
			s.Rename(newName)
		}

		s.Payload = a.ExportableName(s.Payload)

		// fix required trait names
		for i, n := range s.Required {
			s.Required[i] = a.ExportableName(n)
		}
	}

	for _, s := range a.Shapes {
		// fix enum names
		if s.IsEnum() {
			s.EnumConsts = make([]string, len(s.Enum))
			for i := range s.Enum {
				shape := s.ShapeName
				shape = strings.ToUpper(shape[0:1]) + shape[1:]
				s.EnumConsts[i] = shape + s.EnumName(i)
			}
		}
	}
}

// createInputOutputShapes creates toplevel input/output shapes if they
// have not been defined in the API. This normalizes all APIs to always
// have an input and output structure in the signature.
func (a *API) createInputOutputShapes() {
	for _, op := range a.Operations {
		if !op.HasInput() {
			setAsPlacholderShape(&op.InputRef, op.ExportedName+"Input", a)
		}
		if !op.HasOutput() {
			setAsPlacholderShape(&op.OutputRef, op.ExportedName+"Output", a)
		}
	}
}

func setAsPlacholderShape(tgtShapeRef *ShapeRef, name string, a *API) {
	shape := a.makeIOShape(name)
	shape.Placeholder = true
	*tgtShapeRef = ShapeRef{API: a, ShapeName: shape.ShapeName, Shape: shape}
	shape.refs = append(shape.refs, tgtShapeRef)
}

// makeIOShape returns a pointer to a new Shape initialized by the name provided.
func (a *API) makeIOShape(name string) *Shape {
	shape := &Shape{
		API: a, ShapeName: name, Type: "structure",
		MemberRefs: map[string]*ShapeRef{},
	}
	a.Shapes[name] = shape
	return shape
}

// removeUnusedShapes removes shapes from the API which are not referenced by any
// other shape in the API.
func (a *API) removeUnusedShapes() {
	for n, s := range a.Shapes {
		if len(s.refs) == 0 {
			delete(a.Shapes, n)
		}
	}
}

// Represents the service package name to EndpointsID mapping
var custEndpointsKey = map[string]string{
	"applicationautoscaling": "application-autoscaling",
}

// Sents the EndpointsID field of Metadata  with the value of the
// EndpointPrefix if EndpointsID is not set. Also adds
// customizations for services if EndpointPrefix is not a valid key.
func (a *API) setMetadataEndpointsKey() {
	if len(a.Metadata.EndpointsID) != 0 {
		return
	}

	if v, ok := custEndpointsKey[a.PackageName()]; ok {
		a.Metadata.EndpointsID = v
	} else {
		a.Metadata.EndpointsID = a.Metadata.EndpointPrefix
	}
}
