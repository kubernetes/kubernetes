package fieldpath

import (
	"strings"

	"github.com/containerd/containerd/protobuf/plugin"
	"github.com/gogo/protobuf/gogoproto"
	"github.com/gogo/protobuf/protoc-gen-gogo/descriptor"
	"github.com/gogo/protobuf/protoc-gen-gogo/generator"
)

type fieldpathGenerator struct {
	*generator.Generator
	generator.PluginImports
	typeurlPkg generator.Single
}

func init() {
	generator.RegisterPlugin(new(fieldpathGenerator))
}

func (p *fieldpathGenerator) Name() string {
	return "fieldpath"
}

func (p *fieldpathGenerator) Init(g *generator.Generator) {
	p.Generator = g
}

func (p *fieldpathGenerator) Generate(file *generator.FileDescriptor) {
	p.PluginImports = generator.NewPluginImports(p.Generator)
	p.typeurlPkg = p.NewImport("github.com/containerd/typeurl")

	for _, m := range file.Messages() {
		if m.DescriptorProto.GetOptions().GetMapEntry() {
			continue
		}
		if plugin.FieldpathEnabled(file.FileDescriptorProto, m.DescriptorProto) {
			p.generateMessage(m)
		}
	}
}

func (p *fieldpathGenerator) generateMessage(m *generator.Descriptor) {
	ccTypeName := generator.CamelCaseSlice(m.TypeName())

	p.P()
	p.P(`// Field returns the value for the given fieldpath as a string, if defined.`)
	p.P(`// If the value is not defined, the second value will be false.`)
	p.P("func(m *", ccTypeName, ") Field(fieldpath []string) (string, bool) {")
	p.In()

	var (
		fields    []*descriptor.FieldDescriptorProto
		unhandled []*descriptor.FieldDescriptorProto
	)

	for _, f := range m.Field {
		if f.IsBool() || f.IsString() || isLabelsField(f) || isAnyField(f) || isMessageField(f) {
			fields = append(fields, f)
		} else {
			unhandled = append(unhandled, f)
		}
	}

	if len(fields) > 0 {
		p.P(`if len(fieldpath) == 0 {`)
		p.In()
		p.P(`return "", false`)
		p.Out()
		p.P(`}`)
		p.P()
		p.P("switch fieldpath[0] {")

		for _, f := range unhandled {
			p.P("// unhandled: ", f.GetName())
		}

		for _, f := range fields {
			fName := generator.CamelCase(*f.Name)
			if gogoproto.IsCustomName(f) {
				fName = gogoproto.GetCustomName(f)
			}

			p.P(`case "`, f.GetName(), `":`)
			p.In()
			switch {
			case isLabelsField(f):
				p.P(`// Labels fields have been special-cased by name. If this breaks,`)
				p.P(`// add better special casing to fieldpath plugin.`)
				p.P(`if len(m.`, fName, `) == 0 {`)
				p.In()
				p.P(`return "", false`)
				p.Out()
				p.P("}")
				p.P(`value, ok := m.`, fName, `[strings.Join(fieldpath[1:], ".")]`)
				p.P(`return value, ok`)
			case isAnyField(f):
				p.P(`decoded, err := `, p.typeurlPkg.Use(), `.UnmarshalAny(m.`, fName, `)`)
				p.P(`if err != nil {`)
				p.In()
				p.P(`return "", false`)
				p.Out()
				p.P(`}`)
				p.P()
				p.P(`adaptor, ok := decoded.(interface { Field([]string) (string, bool) })`)
				p.P(`if !ok {`)
				p.In()
				p.P(`return "", false`)
				p.Out()
				p.P(`}`)
				p.P(`return adaptor.Field(fieldpath[1:])`)
			case isMessageField(f):
				p.P(`// NOTE(stevvooe): This is probably not correct in many cases.`)
				p.P(`// We assume that the target message also implements the Field`)
				p.P(`// method, which isn't likely true in a lot of cases.`)
				p.P(`//`)
				p.P(`// If you have a broken build and have found this comment,`)
				p.P(`// you may be closer to a solution.`)
				p.P(`if m.`, fName, ` == nil {`)
				p.In()
				p.P(`return "", false`)
				p.Out()
				p.P(`}`)
				p.P()
				p.P(`return m.`, fName, `.Field(fieldpath[1:])`)
			case f.IsString():
				p.P(`return string(m.`, fName, `), len(m.`, fName, `) > 0`)
			case f.IsBool():
				p.P(`return fmt.Sprint(m.`, fName, `), true`)
			}
			p.Out()
		}

		p.P(`}`)
	} else {
		for _, f := range unhandled {
			p.P("// unhandled: ", f.GetName())
		}
	}

	p.P(`return "", false`)
	p.Out()
	p.P("}")
}

func isMessageField(f *descriptor.FieldDescriptorProto) bool {
	return !f.IsRepeated() && f.IsMessage() && f.GetTypeName() != ".google.protobuf.Timestamp"
}

func isLabelsField(f *descriptor.FieldDescriptorProto) bool {
	return f.IsMessage() && f.GetName() == "labels" && strings.HasSuffix(f.GetTypeName(), ".LabelsEntry")
}

func isAnyField(f *descriptor.FieldDescriptorProto) bool {
	return !f.IsRepeated() && f.IsMessage() && f.GetTypeName() == ".google.protobuf.Any"
}
