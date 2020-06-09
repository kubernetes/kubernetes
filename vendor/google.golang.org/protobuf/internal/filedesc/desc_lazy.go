// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package filedesc

import (
	"reflect"
	"sync"

	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/internal/descopts"
	"google.golang.org/protobuf/internal/fieldnum"
	"google.golang.org/protobuf/internal/strs"
	"google.golang.org/protobuf/proto"
	pref "google.golang.org/protobuf/reflect/protoreflect"
)

func (fd *File) lazyRawInit() {
	fd.unmarshalFull(fd.builder.RawDescriptor)
	fd.resolveMessages()
	fd.resolveExtensions()
	fd.resolveServices()
}

func (file *File) resolveMessages() {
	var depIdx int32
	for i := range file.allMessages {
		md := &file.allMessages[i]

		// Resolve message field dependencies.
		for j := range md.L2.Fields.List {
			fd := &md.L2.Fields.List[j]

			// Weak fields are resolved upon actual use.
			if fd.L1.IsWeak {
				continue
			}

			// Resolve message field dependency.
			switch fd.L1.Kind {
			case pref.EnumKind:
				fd.L1.Enum = file.resolveEnumDependency(fd.L1.Enum, listFieldDeps, depIdx)
				depIdx++
			case pref.MessageKind, pref.GroupKind:
				fd.L1.Message = file.resolveMessageDependency(fd.L1.Message, listFieldDeps, depIdx)
				depIdx++
			}

			// Default is resolved here since it depends on Enum being resolved.
			if v := fd.L1.Default.val; v.IsValid() {
				fd.L1.Default = unmarshalDefault(v.Bytes(), fd.L1.Kind, file, fd.L1.Enum)
			}
		}
	}
}

func (file *File) resolveExtensions() {
	var depIdx int32
	for i := range file.allExtensions {
		xd := &file.allExtensions[i]

		// Resolve extension field dependency.
		switch xd.L1.Kind {
		case pref.EnumKind:
			xd.L2.Enum = file.resolveEnumDependency(xd.L2.Enum, listExtDeps, depIdx)
			depIdx++
		case pref.MessageKind, pref.GroupKind:
			xd.L2.Message = file.resolveMessageDependency(xd.L2.Message, listExtDeps, depIdx)
			depIdx++
		}

		// Default is resolved here since it depends on Enum being resolved.
		if v := xd.L2.Default.val; v.IsValid() {
			xd.L2.Default = unmarshalDefault(v.Bytes(), xd.L1.Kind, file, xd.L2.Enum)
		}
	}
}

func (file *File) resolveServices() {
	var depIdx int32
	for i := range file.allServices {
		sd := &file.allServices[i]

		// Resolve method dependencies.
		for j := range sd.L2.Methods.List {
			md := &sd.L2.Methods.List[j]
			md.L1.Input = file.resolveMessageDependency(md.L1.Input, listMethInDeps, depIdx)
			md.L1.Output = file.resolveMessageDependency(md.L1.Output, listMethOutDeps, depIdx)
			depIdx++
		}
	}
}

func (file *File) resolveEnumDependency(ed pref.EnumDescriptor, i, j int32) pref.EnumDescriptor {
	r := file.builder.FileRegistry
	if r, ok := r.(resolverByIndex); ok {
		if ed2 := r.FindEnumByIndex(i, j, file.allEnums, file.allMessages); ed2 != nil {
			return ed2
		}
	}
	for i := range file.allEnums {
		if ed2 := &file.allEnums[i]; ed2.L0.FullName == ed.FullName() {
			return ed2
		}
	}
	if d, _ := r.FindDescriptorByName(ed.FullName()); d != nil {
		return d.(pref.EnumDescriptor)
	}
	return ed
}

func (file *File) resolveMessageDependency(md pref.MessageDescriptor, i, j int32) pref.MessageDescriptor {
	r := file.builder.FileRegistry
	if r, ok := r.(resolverByIndex); ok {
		if md2 := r.FindMessageByIndex(i, j, file.allEnums, file.allMessages); md2 != nil {
			return md2
		}
	}
	for i := range file.allMessages {
		if md2 := &file.allMessages[i]; md2.L0.FullName == md.FullName() {
			return md2
		}
	}
	if d, _ := r.FindDescriptorByName(md.FullName()); d != nil {
		return d.(pref.MessageDescriptor)
	}
	return md
}

func (fd *File) unmarshalFull(b []byte) {
	sb := getBuilder()
	defer putBuilder(sb)

	var enumIdx, messageIdx, extensionIdx, serviceIdx int
	var rawOptions []byte
	fd.L2 = new(FileL2)
	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		b = b[n:]
		switch typ {
		case protowire.VarintType:
			v, m := protowire.ConsumeVarint(b)
			b = b[m:]
			switch num {
			case fieldnum.FileDescriptorProto_PublicDependency:
				fd.L2.Imports[v].IsPublic = true
			case fieldnum.FileDescriptorProto_WeakDependency:
				fd.L2.Imports[v].IsWeak = true
			}
		case protowire.BytesType:
			v, m := protowire.ConsumeBytes(b)
			b = b[m:]
			switch num {
			case fieldnum.FileDescriptorProto_Dependency:
				path := sb.MakeString(v)
				imp, _ := fd.builder.FileRegistry.FindFileByPath(path)
				if imp == nil {
					imp = PlaceholderFile(path)
				}
				fd.L2.Imports = append(fd.L2.Imports, pref.FileImport{FileDescriptor: imp})
			case fieldnum.FileDescriptorProto_EnumType:
				fd.L1.Enums.List[enumIdx].unmarshalFull(v, sb)
				enumIdx++
			case fieldnum.FileDescriptorProto_MessageType:
				fd.L1.Messages.List[messageIdx].unmarshalFull(v, sb)
				messageIdx++
			case fieldnum.FileDescriptorProto_Extension:
				fd.L1.Extensions.List[extensionIdx].unmarshalFull(v, sb)
				extensionIdx++
			case fieldnum.FileDescriptorProto_Service:
				fd.L1.Services.List[serviceIdx].unmarshalFull(v, sb)
				serviceIdx++
			case fieldnum.FileDescriptorProto_Options:
				rawOptions = appendOptions(rawOptions, v)
			}
		default:
			m := protowire.ConsumeFieldValue(num, typ, b)
			b = b[m:]
		}
	}
	fd.L2.Options = fd.builder.optionsUnmarshaler(&descopts.File, rawOptions)
}

func (ed *Enum) unmarshalFull(b []byte, sb *strs.Builder) {
	var rawValues [][]byte
	var rawOptions []byte
	if !ed.L1.eagerValues {
		ed.L2 = new(EnumL2)
	}
	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		b = b[n:]
		switch typ {
		case protowire.BytesType:
			v, m := protowire.ConsumeBytes(b)
			b = b[m:]
			switch num {
			case fieldnum.EnumDescriptorProto_Value:
				rawValues = append(rawValues, v)
			case fieldnum.EnumDescriptorProto_ReservedName:
				ed.L2.ReservedNames.List = append(ed.L2.ReservedNames.List, pref.Name(sb.MakeString(v)))
			case fieldnum.EnumDescriptorProto_ReservedRange:
				ed.L2.ReservedRanges.List = append(ed.L2.ReservedRanges.List, unmarshalEnumReservedRange(v))
			case fieldnum.EnumDescriptorProto_Options:
				rawOptions = appendOptions(rawOptions, v)
			}
		default:
			m := protowire.ConsumeFieldValue(num, typ, b)
			b = b[m:]
		}
	}
	if !ed.L1.eagerValues && len(rawValues) > 0 {
		ed.L2.Values.List = make([]EnumValue, len(rawValues))
		for i, b := range rawValues {
			ed.L2.Values.List[i].unmarshalFull(b, sb, ed.L0.ParentFile, ed, i)
		}
	}
	ed.L2.Options = ed.L0.ParentFile.builder.optionsUnmarshaler(&descopts.Enum, rawOptions)
}

func unmarshalEnumReservedRange(b []byte) (r [2]pref.EnumNumber) {
	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		b = b[n:]
		switch typ {
		case protowire.VarintType:
			v, m := protowire.ConsumeVarint(b)
			b = b[m:]
			switch num {
			case fieldnum.EnumDescriptorProto_EnumReservedRange_Start:
				r[0] = pref.EnumNumber(v)
			case fieldnum.EnumDescriptorProto_EnumReservedRange_End:
				r[1] = pref.EnumNumber(v)
			}
		default:
			m := protowire.ConsumeFieldValue(num, typ, b)
			b = b[m:]
		}
	}
	return r
}

func (vd *EnumValue) unmarshalFull(b []byte, sb *strs.Builder, pf *File, pd pref.Descriptor, i int) {
	vd.L0.ParentFile = pf
	vd.L0.Parent = pd
	vd.L0.Index = i

	var rawOptions []byte
	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		b = b[n:]
		switch typ {
		case protowire.VarintType:
			v, m := protowire.ConsumeVarint(b)
			b = b[m:]
			switch num {
			case fieldnum.EnumValueDescriptorProto_Number:
				vd.L1.Number = pref.EnumNumber(v)
			}
		case protowire.BytesType:
			v, m := protowire.ConsumeBytes(b)
			b = b[m:]
			switch num {
			case fieldnum.EnumValueDescriptorProto_Name:
				// NOTE: Enum values are in the same scope as the enum parent.
				vd.L0.FullName = appendFullName(sb, pd.Parent().FullName(), v)
			case fieldnum.EnumValueDescriptorProto_Options:
				rawOptions = appendOptions(rawOptions, v)
			}
		default:
			m := protowire.ConsumeFieldValue(num, typ, b)
			b = b[m:]
		}
	}
	vd.L1.Options = pf.builder.optionsUnmarshaler(&descopts.EnumValue, rawOptions)
}

func (md *Message) unmarshalFull(b []byte, sb *strs.Builder) {
	var rawFields, rawOneofs [][]byte
	var enumIdx, messageIdx, extensionIdx int
	var rawOptions []byte
	md.L2 = new(MessageL2)
	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		b = b[n:]
		switch typ {
		case protowire.BytesType:
			v, m := protowire.ConsumeBytes(b)
			b = b[m:]
			switch num {
			case fieldnum.DescriptorProto_Field:
				rawFields = append(rawFields, v)
			case fieldnum.DescriptorProto_OneofDecl:
				rawOneofs = append(rawOneofs, v)
			case fieldnum.DescriptorProto_ReservedName:
				md.L2.ReservedNames.List = append(md.L2.ReservedNames.List, pref.Name(sb.MakeString(v)))
			case fieldnum.DescriptorProto_ReservedRange:
				md.L2.ReservedRanges.List = append(md.L2.ReservedRanges.List, unmarshalMessageReservedRange(v))
			case fieldnum.DescriptorProto_ExtensionRange:
				r, rawOptions := unmarshalMessageExtensionRange(v)
				opts := md.L0.ParentFile.builder.optionsUnmarshaler(&descopts.ExtensionRange, rawOptions)
				md.L2.ExtensionRanges.List = append(md.L2.ExtensionRanges.List, r)
				md.L2.ExtensionRangeOptions = append(md.L2.ExtensionRangeOptions, opts)
			case fieldnum.DescriptorProto_EnumType:
				md.L1.Enums.List[enumIdx].unmarshalFull(v, sb)
				enumIdx++
			case fieldnum.DescriptorProto_NestedType:
				md.L1.Messages.List[messageIdx].unmarshalFull(v, sb)
				messageIdx++
			case fieldnum.DescriptorProto_Extension:
				md.L1.Extensions.List[extensionIdx].unmarshalFull(v, sb)
				extensionIdx++
			case fieldnum.DescriptorProto_Options:
				md.unmarshalOptions(v)
				rawOptions = appendOptions(rawOptions, v)
			}
		default:
			m := protowire.ConsumeFieldValue(num, typ, b)
			b = b[m:]
		}
	}
	if len(rawFields) > 0 || len(rawOneofs) > 0 {
		md.L2.Fields.List = make([]Field, len(rawFields))
		md.L2.Oneofs.List = make([]Oneof, len(rawOneofs))
		for i, b := range rawFields {
			fd := &md.L2.Fields.List[i]
			fd.unmarshalFull(b, sb, md.L0.ParentFile, md, i)
			if fd.L1.Cardinality == pref.Required {
				md.L2.RequiredNumbers.List = append(md.L2.RequiredNumbers.List, fd.L1.Number)
			}
		}
		for i, b := range rawOneofs {
			od := &md.L2.Oneofs.List[i]
			od.unmarshalFull(b, sb, md.L0.ParentFile, md, i)
		}
	}
	md.L2.Options = md.L0.ParentFile.builder.optionsUnmarshaler(&descopts.Message, rawOptions)
}

func (md *Message) unmarshalOptions(b []byte) {
	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		b = b[n:]
		switch typ {
		case protowire.VarintType:
			v, m := protowire.ConsumeVarint(b)
			b = b[m:]
			switch num {
			case fieldnum.MessageOptions_MapEntry:
				md.L1.IsMapEntry = protowire.DecodeBool(v)
			case fieldnum.MessageOptions_MessageSetWireFormat:
				md.L1.IsMessageSet = protowire.DecodeBool(v)
			}
		default:
			m := protowire.ConsumeFieldValue(num, typ, b)
			b = b[m:]
		}
	}
}

func unmarshalMessageReservedRange(b []byte) (r [2]pref.FieldNumber) {
	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		b = b[n:]
		switch typ {
		case protowire.VarintType:
			v, m := protowire.ConsumeVarint(b)
			b = b[m:]
			switch num {
			case fieldnum.DescriptorProto_ReservedRange_Start:
				r[0] = pref.FieldNumber(v)
			case fieldnum.DescriptorProto_ReservedRange_End:
				r[1] = pref.FieldNumber(v)
			}
		default:
			m := protowire.ConsumeFieldValue(num, typ, b)
			b = b[m:]
		}
	}
	return r
}

func unmarshalMessageExtensionRange(b []byte) (r [2]pref.FieldNumber, rawOptions []byte) {
	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		b = b[n:]
		switch typ {
		case protowire.VarintType:
			v, m := protowire.ConsumeVarint(b)
			b = b[m:]
			switch num {
			case fieldnum.DescriptorProto_ExtensionRange_Start:
				r[0] = pref.FieldNumber(v)
			case fieldnum.DescriptorProto_ExtensionRange_End:
				r[1] = pref.FieldNumber(v)
			}
		case protowire.BytesType:
			v, m := protowire.ConsumeBytes(b)
			b = b[m:]
			switch num {
			case fieldnum.DescriptorProto_ExtensionRange_Options:
				rawOptions = appendOptions(rawOptions, v)
			}
		default:
			m := protowire.ConsumeFieldValue(num, typ, b)
			b = b[m:]
		}
	}
	return r, rawOptions
}

func (fd *Field) unmarshalFull(b []byte, sb *strs.Builder, pf *File, pd pref.Descriptor, i int) {
	fd.L0.ParentFile = pf
	fd.L0.Parent = pd
	fd.L0.Index = i

	var rawTypeName []byte
	var rawOptions []byte
	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		b = b[n:]
		switch typ {
		case protowire.VarintType:
			v, m := protowire.ConsumeVarint(b)
			b = b[m:]
			switch num {
			case fieldnum.FieldDescriptorProto_Number:
				fd.L1.Number = pref.FieldNumber(v)
			case fieldnum.FieldDescriptorProto_Label:
				fd.L1.Cardinality = pref.Cardinality(v)
			case fieldnum.FieldDescriptorProto_Type:
				fd.L1.Kind = pref.Kind(v)
			case fieldnum.FieldDescriptorProto_OneofIndex:
				// In Message.unmarshalFull, we allocate slices for both
				// the field and oneof descriptors before unmarshaling either
				// of them. This ensures pointers to slice elements are stable.
				od := &pd.(*Message).L2.Oneofs.List[v]
				od.L1.Fields.List = append(od.L1.Fields.List, fd)
				if fd.L1.ContainingOneof != nil {
					panic("oneof type already set")
				}
				fd.L1.ContainingOneof = od
			case fieldnum.FieldDescriptorProto_Proto3Optional:
				fd.L1.IsProto3Optional = protowire.DecodeBool(v)
			}
		case protowire.BytesType:
			v, m := protowire.ConsumeBytes(b)
			b = b[m:]
			switch num {
			case fieldnum.FieldDescriptorProto_Name:
				fd.L0.FullName = appendFullName(sb, pd.FullName(), v)
			case fieldnum.FieldDescriptorProto_JsonName:
				fd.L1.JSONName.Init(sb.MakeString(v))
			case fieldnum.FieldDescriptorProto_DefaultValue:
				fd.L1.Default.val = pref.ValueOfBytes(v) // temporarily store as bytes; later resolved in resolveMessages
			case fieldnum.FieldDescriptorProto_TypeName:
				rawTypeName = v
			case fieldnum.FieldDescriptorProto_Options:
				fd.unmarshalOptions(v)
				rawOptions = appendOptions(rawOptions, v)
			}
		default:
			m := protowire.ConsumeFieldValue(num, typ, b)
			b = b[m:]
		}
	}
	if rawTypeName != nil {
		name := makeFullName(sb, rawTypeName)
		switch fd.L1.Kind {
		case pref.EnumKind:
			fd.L1.Enum = PlaceholderEnum(name)
		case pref.MessageKind, pref.GroupKind:
			fd.L1.Message = PlaceholderMessage(name)
		}
	}
	fd.L1.Options = pf.builder.optionsUnmarshaler(&descopts.Field, rawOptions)
}

func (fd *Field) unmarshalOptions(b []byte) {
	const FieldOptions_EnforceUTF8 = 13

	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		b = b[n:]
		switch typ {
		case protowire.VarintType:
			v, m := protowire.ConsumeVarint(b)
			b = b[m:]
			switch num {
			case fieldnum.FieldOptions_Packed:
				fd.L1.HasPacked = true
				fd.L1.IsPacked = protowire.DecodeBool(v)
			case fieldnum.FieldOptions_Weak:
				fd.L1.IsWeak = protowire.DecodeBool(v)
			case FieldOptions_EnforceUTF8:
				fd.L1.HasEnforceUTF8 = true
				fd.L1.EnforceUTF8 = protowire.DecodeBool(v)
			}
		default:
			m := protowire.ConsumeFieldValue(num, typ, b)
			b = b[m:]
		}
	}
}

func (od *Oneof) unmarshalFull(b []byte, sb *strs.Builder, pf *File, pd pref.Descriptor, i int) {
	od.L0.ParentFile = pf
	od.L0.Parent = pd
	od.L0.Index = i

	var rawOptions []byte
	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		b = b[n:]
		switch typ {
		case protowire.BytesType:
			v, m := protowire.ConsumeBytes(b)
			b = b[m:]
			switch num {
			case fieldnum.OneofDescriptorProto_Name:
				od.L0.FullName = appendFullName(sb, pd.FullName(), v)
			case fieldnum.OneofDescriptorProto_Options:
				rawOptions = appendOptions(rawOptions, v)
			}
		default:
			m := protowire.ConsumeFieldValue(num, typ, b)
			b = b[m:]
		}
	}
	od.L1.Options = pf.builder.optionsUnmarshaler(&descopts.Oneof, rawOptions)
}

func (xd *Extension) unmarshalFull(b []byte, sb *strs.Builder) {
	var rawTypeName []byte
	var rawOptions []byte
	xd.L2 = new(ExtensionL2)
	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		b = b[n:]
		switch typ {
		case protowire.VarintType:
			v, m := protowire.ConsumeVarint(b)
			b = b[m:]
			switch num {
			case fieldnum.FieldDescriptorProto_Proto3Optional:
				xd.L2.IsProto3Optional = protowire.DecodeBool(v)
			}
		case protowire.BytesType:
			v, m := protowire.ConsumeBytes(b)
			b = b[m:]
			switch num {
			case fieldnum.FieldDescriptorProto_JsonName:
				xd.L2.JSONName.Init(sb.MakeString(v))
			case fieldnum.FieldDescriptorProto_DefaultValue:
				xd.L2.Default.val = pref.ValueOfBytes(v) // temporarily store as bytes; later resolved in resolveExtensions
			case fieldnum.FieldDescriptorProto_TypeName:
				rawTypeName = v
			case fieldnum.FieldDescriptorProto_Options:
				xd.unmarshalOptions(v)
				rawOptions = appendOptions(rawOptions, v)
			}
		default:
			m := protowire.ConsumeFieldValue(num, typ, b)
			b = b[m:]
		}
	}
	if rawTypeName != nil {
		name := makeFullName(sb, rawTypeName)
		switch xd.L1.Kind {
		case pref.EnumKind:
			xd.L2.Enum = PlaceholderEnum(name)
		case pref.MessageKind, pref.GroupKind:
			xd.L2.Message = PlaceholderMessage(name)
		}
	}
	xd.L2.Options = xd.L0.ParentFile.builder.optionsUnmarshaler(&descopts.Field, rawOptions)
}

func (xd *Extension) unmarshalOptions(b []byte) {
	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		b = b[n:]
		switch typ {
		case protowire.VarintType:
			v, m := protowire.ConsumeVarint(b)
			b = b[m:]
			switch num {
			case fieldnum.FieldOptions_Packed:
				xd.L2.IsPacked = protowire.DecodeBool(v)
			}
		default:
			m := protowire.ConsumeFieldValue(num, typ, b)
			b = b[m:]
		}
	}
}

func (sd *Service) unmarshalFull(b []byte, sb *strs.Builder) {
	var rawMethods [][]byte
	var rawOptions []byte
	sd.L2 = new(ServiceL2)
	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		b = b[n:]
		switch typ {
		case protowire.BytesType:
			v, m := protowire.ConsumeBytes(b)
			b = b[m:]
			switch num {
			case fieldnum.ServiceDescriptorProto_Method:
				rawMethods = append(rawMethods, v)
			case fieldnum.ServiceDescriptorProto_Options:
				rawOptions = appendOptions(rawOptions, v)
			}
		default:
			m := protowire.ConsumeFieldValue(num, typ, b)
			b = b[m:]
		}
	}
	if len(rawMethods) > 0 {
		sd.L2.Methods.List = make([]Method, len(rawMethods))
		for i, b := range rawMethods {
			sd.L2.Methods.List[i].unmarshalFull(b, sb, sd.L0.ParentFile, sd, i)
		}
	}
	sd.L2.Options = sd.L0.ParentFile.builder.optionsUnmarshaler(&descopts.Service, rawOptions)
}

func (md *Method) unmarshalFull(b []byte, sb *strs.Builder, pf *File, pd pref.Descriptor, i int) {
	md.L0.ParentFile = pf
	md.L0.Parent = pd
	md.L0.Index = i

	var rawOptions []byte
	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		b = b[n:]
		switch typ {
		case protowire.VarintType:
			v, m := protowire.ConsumeVarint(b)
			b = b[m:]
			switch num {
			case fieldnum.MethodDescriptorProto_ClientStreaming:
				md.L1.IsStreamingClient = protowire.DecodeBool(v)
			case fieldnum.MethodDescriptorProto_ServerStreaming:
				md.L1.IsStreamingServer = protowire.DecodeBool(v)
			}
		case protowire.BytesType:
			v, m := protowire.ConsumeBytes(b)
			b = b[m:]
			switch num {
			case fieldnum.MethodDescriptorProto_Name:
				md.L0.FullName = appendFullName(sb, pd.FullName(), v)
			case fieldnum.MethodDescriptorProto_InputType:
				md.L1.Input = PlaceholderMessage(makeFullName(sb, v))
			case fieldnum.MethodDescriptorProto_OutputType:
				md.L1.Output = PlaceholderMessage(makeFullName(sb, v))
			case fieldnum.MethodDescriptorProto_Options:
				rawOptions = appendOptions(rawOptions, v)
			}
		default:
			m := protowire.ConsumeFieldValue(num, typ, b)
			b = b[m:]
		}
	}
	md.L1.Options = pf.builder.optionsUnmarshaler(&descopts.Method, rawOptions)
}

// appendOptions appends src to dst, where the returned slice is never nil.
// This is necessary to distinguish between empty and unpopulated options.
func appendOptions(dst, src []byte) []byte {
	if dst == nil {
		dst = []byte{}
	}
	return append(dst, src...)
}

// optionsUnmarshaler constructs a lazy unmarshal function for an options message.
//
// The type of message to unmarshal to is passed as a pointer since the
// vars in descopts may not yet be populated at the time this function is called.
func (db *Builder) optionsUnmarshaler(p *pref.ProtoMessage, b []byte) func() pref.ProtoMessage {
	if b == nil {
		return nil
	}
	var opts pref.ProtoMessage
	var once sync.Once
	return func() pref.ProtoMessage {
		once.Do(func() {
			if *p == nil {
				panic("Descriptor.Options called without importing the descriptor package")
			}
			opts = reflect.New(reflect.TypeOf(*p).Elem()).Interface().(pref.ProtoMessage)
			if err := (proto.UnmarshalOptions{
				AllowPartial: true,
				Resolver:     db.TypeResolver,
			}).Unmarshal(b, opts); err != nil {
				panic(err)
			}
		})
		return opts
	}
}
