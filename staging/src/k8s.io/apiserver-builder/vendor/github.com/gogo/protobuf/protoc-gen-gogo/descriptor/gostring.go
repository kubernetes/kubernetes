package descriptor

import fmt "fmt"

import strings "strings"
import github_com_gogo_protobuf_proto "github.com/gogo/protobuf/proto"
import sort "sort"
import strconv "strconv"
import reflect "reflect"

func (this *FileDescriptorSet) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 5)
	s = append(s, "&descriptor.FileDescriptorSet{")
	if this.File != nil {
		s = append(s, "File: "+fmt.Sprintf("%#v", this.File)+",\n")
	}
	if this.XXX_unrecognized != nil {
		s = append(s, "XXX_unrecognized:"+fmt.Sprintf("%#v", this.XXX_unrecognized)+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func (this *FileDescriptorProto) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 16)
	s = append(s, "&descriptor.FileDescriptorProto{")
	if this.Name != nil {
		s = append(s, "Name: "+valueToGoStringDescriptor(this.Name, "string")+",\n")
	}
	if this.Package != nil {
		s = append(s, "Package: "+valueToGoStringDescriptor(this.Package, "string")+",\n")
	}
	if this.Dependency != nil {
		s = append(s, "Dependency: "+fmt.Sprintf("%#v", this.Dependency)+",\n")
	}
	if this.PublicDependency != nil {
		s = append(s, "PublicDependency: "+fmt.Sprintf("%#v", this.PublicDependency)+",\n")
	}
	if this.WeakDependency != nil {
		s = append(s, "WeakDependency: "+fmt.Sprintf("%#v", this.WeakDependency)+",\n")
	}
	if this.MessageType != nil {
		s = append(s, "MessageType: "+fmt.Sprintf("%#v", this.MessageType)+",\n")
	}
	if this.EnumType != nil {
		s = append(s, "EnumType: "+fmt.Sprintf("%#v", this.EnumType)+",\n")
	}
	if this.Service != nil {
		s = append(s, "Service: "+fmt.Sprintf("%#v", this.Service)+",\n")
	}
	if this.Extension != nil {
		s = append(s, "Extension: "+fmt.Sprintf("%#v", this.Extension)+",\n")
	}
	if this.Options != nil {
		s = append(s, "Options: "+fmt.Sprintf("%#v", this.Options)+",\n")
	}
	if this.SourceCodeInfo != nil {
		s = append(s, "SourceCodeInfo: "+fmt.Sprintf("%#v", this.SourceCodeInfo)+",\n")
	}
	if this.Syntax != nil {
		s = append(s, "Syntax: "+valueToGoStringDescriptor(this.Syntax, "string")+",\n")
	}
	if this.XXX_unrecognized != nil {
		s = append(s, "XXX_unrecognized:"+fmt.Sprintf("%#v", this.XXX_unrecognized)+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func (this *DescriptorProto) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 14)
	s = append(s, "&descriptor.DescriptorProto{")
	if this.Name != nil {
		s = append(s, "Name: "+valueToGoStringDescriptor(this.Name, "string")+",\n")
	}
	if this.Field != nil {
		s = append(s, "Field: "+fmt.Sprintf("%#v", this.Field)+",\n")
	}
	if this.Extension != nil {
		s = append(s, "Extension: "+fmt.Sprintf("%#v", this.Extension)+",\n")
	}
	if this.NestedType != nil {
		s = append(s, "NestedType: "+fmt.Sprintf("%#v", this.NestedType)+",\n")
	}
	if this.EnumType != nil {
		s = append(s, "EnumType: "+fmt.Sprintf("%#v", this.EnumType)+",\n")
	}
	if this.ExtensionRange != nil {
		s = append(s, "ExtensionRange: "+fmt.Sprintf("%#v", this.ExtensionRange)+",\n")
	}
	if this.OneofDecl != nil {
		s = append(s, "OneofDecl: "+fmt.Sprintf("%#v", this.OneofDecl)+",\n")
	}
	if this.Options != nil {
		s = append(s, "Options: "+fmt.Sprintf("%#v", this.Options)+",\n")
	}
	if this.ReservedRange != nil {
		s = append(s, "ReservedRange: "+fmt.Sprintf("%#v", this.ReservedRange)+",\n")
	}
	if this.ReservedName != nil {
		s = append(s, "ReservedName: "+fmt.Sprintf("%#v", this.ReservedName)+",\n")
	}
	if this.XXX_unrecognized != nil {
		s = append(s, "XXX_unrecognized:"+fmt.Sprintf("%#v", this.XXX_unrecognized)+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func (this *DescriptorProto_ExtensionRange) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 6)
	s = append(s, "&descriptor.DescriptorProto_ExtensionRange{")
	if this.Start != nil {
		s = append(s, "Start: "+valueToGoStringDescriptor(this.Start, "int32")+",\n")
	}
	if this.End != nil {
		s = append(s, "End: "+valueToGoStringDescriptor(this.End, "int32")+",\n")
	}
	if this.XXX_unrecognized != nil {
		s = append(s, "XXX_unrecognized:"+fmt.Sprintf("%#v", this.XXX_unrecognized)+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func (this *DescriptorProto_ReservedRange) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 6)
	s = append(s, "&descriptor.DescriptorProto_ReservedRange{")
	if this.Start != nil {
		s = append(s, "Start: "+valueToGoStringDescriptor(this.Start, "int32")+",\n")
	}
	if this.End != nil {
		s = append(s, "End: "+valueToGoStringDescriptor(this.End, "int32")+",\n")
	}
	if this.XXX_unrecognized != nil {
		s = append(s, "XXX_unrecognized:"+fmt.Sprintf("%#v", this.XXX_unrecognized)+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func (this *FieldDescriptorProto) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 14)
	s = append(s, "&descriptor.FieldDescriptorProto{")
	if this.Name != nil {
		s = append(s, "Name: "+valueToGoStringDescriptor(this.Name, "string")+",\n")
	}
	if this.Number != nil {
		s = append(s, "Number: "+valueToGoStringDescriptor(this.Number, "int32")+",\n")
	}
	if this.Label != nil {
		s = append(s, "Label: "+valueToGoStringDescriptor(this.Label, "descriptor.FieldDescriptorProto_Label")+",\n")
	}
	if this.Type != nil {
		s = append(s, "Type: "+valueToGoStringDescriptor(this.Type, "descriptor.FieldDescriptorProto_Type")+",\n")
	}
	if this.TypeName != nil {
		s = append(s, "TypeName: "+valueToGoStringDescriptor(this.TypeName, "string")+",\n")
	}
	if this.Extendee != nil {
		s = append(s, "Extendee: "+valueToGoStringDescriptor(this.Extendee, "string")+",\n")
	}
	if this.DefaultValue != nil {
		s = append(s, "DefaultValue: "+valueToGoStringDescriptor(this.DefaultValue, "string")+",\n")
	}
	if this.OneofIndex != nil {
		s = append(s, "OneofIndex: "+valueToGoStringDescriptor(this.OneofIndex, "int32")+",\n")
	}
	if this.JsonName != nil {
		s = append(s, "JsonName: "+valueToGoStringDescriptor(this.JsonName, "string")+",\n")
	}
	if this.Options != nil {
		s = append(s, "Options: "+fmt.Sprintf("%#v", this.Options)+",\n")
	}
	if this.XXX_unrecognized != nil {
		s = append(s, "XXX_unrecognized:"+fmt.Sprintf("%#v", this.XXX_unrecognized)+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func (this *OneofDescriptorProto) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 5)
	s = append(s, "&descriptor.OneofDescriptorProto{")
	if this.Name != nil {
		s = append(s, "Name: "+valueToGoStringDescriptor(this.Name, "string")+",\n")
	}
	if this.XXX_unrecognized != nil {
		s = append(s, "XXX_unrecognized:"+fmt.Sprintf("%#v", this.XXX_unrecognized)+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func (this *EnumDescriptorProto) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 7)
	s = append(s, "&descriptor.EnumDescriptorProto{")
	if this.Name != nil {
		s = append(s, "Name: "+valueToGoStringDescriptor(this.Name, "string")+",\n")
	}
	if this.Value != nil {
		s = append(s, "Value: "+fmt.Sprintf("%#v", this.Value)+",\n")
	}
	if this.Options != nil {
		s = append(s, "Options: "+fmt.Sprintf("%#v", this.Options)+",\n")
	}
	if this.XXX_unrecognized != nil {
		s = append(s, "XXX_unrecognized:"+fmt.Sprintf("%#v", this.XXX_unrecognized)+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func (this *EnumValueDescriptorProto) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 7)
	s = append(s, "&descriptor.EnumValueDescriptorProto{")
	if this.Name != nil {
		s = append(s, "Name: "+valueToGoStringDescriptor(this.Name, "string")+",\n")
	}
	if this.Number != nil {
		s = append(s, "Number: "+valueToGoStringDescriptor(this.Number, "int32")+",\n")
	}
	if this.Options != nil {
		s = append(s, "Options: "+fmt.Sprintf("%#v", this.Options)+",\n")
	}
	if this.XXX_unrecognized != nil {
		s = append(s, "XXX_unrecognized:"+fmt.Sprintf("%#v", this.XXX_unrecognized)+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func (this *ServiceDescriptorProto) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 7)
	s = append(s, "&descriptor.ServiceDescriptorProto{")
	if this.Name != nil {
		s = append(s, "Name: "+valueToGoStringDescriptor(this.Name, "string")+",\n")
	}
	if this.Method != nil {
		s = append(s, "Method: "+fmt.Sprintf("%#v", this.Method)+",\n")
	}
	if this.Options != nil {
		s = append(s, "Options: "+fmt.Sprintf("%#v", this.Options)+",\n")
	}
	if this.XXX_unrecognized != nil {
		s = append(s, "XXX_unrecognized:"+fmt.Sprintf("%#v", this.XXX_unrecognized)+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func (this *MethodDescriptorProto) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 10)
	s = append(s, "&descriptor.MethodDescriptorProto{")
	if this.Name != nil {
		s = append(s, "Name: "+valueToGoStringDescriptor(this.Name, "string")+",\n")
	}
	if this.InputType != nil {
		s = append(s, "InputType: "+valueToGoStringDescriptor(this.InputType, "string")+",\n")
	}
	if this.OutputType != nil {
		s = append(s, "OutputType: "+valueToGoStringDescriptor(this.OutputType, "string")+",\n")
	}
	if this.Options != nil {
		s = append(s, "Options: "+fmt.Sprintf("%#v", this.Options)+",\n")
	}
	if this.ClientStreaming != nil {
		s = append(s, "ClientStreaming: "+valueToGoStringDescriptor(this.ClientStreaming, "bool")+",\n")
	}
	if this.ServerStreaming != nil {
		s = append(s, "ServerStreaming: "+valueToGoStringDescriptor(this.ServerStreaming, "bool")+",\n")
	}
	if this.XXX_unrecognized != nil {
		s = append(s, "XXX_unrecognized:"+fmt.Sprintf("%#v", this.XXX_unrecognized)+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func (this *FileOptions) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 20)
	s = append(s, "&descriptor.FileOptions{")
	if this.JavaPackage != nil {
		s = append(s, "JavaPackage: "+valueToGoStringDescriptor(this.JavaPackage, "string")+",\n")
	}
	if this.JavaOuterClassname != nil {
		s = append(s, "JavaOuterClassname: "+valueToGoStringDescriptor(this.JavaOuterClassname, "string")+",\n")
	}
	if this.JavaMultipleFiles != nil {
		s = append(s, "JavaMultipleFiles: "+valueToGoStringDescriptor(this.JavaMultipleFiles, "bool")+",\n")
	}
	if this.JavaGenerateEqualsAndHash != nil {
		s = append(s, "JavaGenerateEqualsAndHash: "+valueToGoStringDescriptor(this.JavaGenerateEqualsAndHash, "bool")+",\n")
	}
	if this.JavaStringCheckUtf8 != nil {
		s = append(s, "JavaStringCheckUtf8: "+valueToGoStringDescriptor(this.JavaStringCheckUtf8, "bool")+",\n")
	}
	if this.OptimizeFor != nil {
		s = append(s, "OptimizeFor: "+valueToGoStringDescriptor(this.OptimizeFor, "descriptor.FileOptions_OptimizeMode")+",\n")
	}
	if this.GoPackage != nil {
		s = append(s, "GoPackage: "+valueToGoStringDescriptor(this.GoPackage, "string")+",\n")
	}
	if this.CcGenericServices != nil {
		s = append(s, "CcGenericServices: "+valueToGoStringDescriptor(this.CcGenericServices, "bool")+",\n")
	}
	if this.JavaGenericServices != nil {
		s = append(s, "JavaGenericServices: "+valueToGoStringDescriptor(this.JavaGenericServices, "bool")+",\n")
	}
	if this.PyGenericServices != nil {
		s = append(s, "PyGenericServices: "+valueToGoStringDescriptor(this.PyGenericServices, "bool")+",\n")
	}
	if this.Deprecated != nil {
		s = append(s, "Deprecated: "+valueToGoStringDescriptor(this.Deprecated, "bool")+",\n")
	}
	if this.CcEnableArenas != nil {
		s = append(s, "CcEnableArenas: "+valueToGoStringDescriptor(this.CcEnableArenas, "bool")+",\n")
	}
	if this.ObjcClassPrefix != nil {
		s = append(s, "ObjcClassPrefix: "+valueToGoStringDescriptor(this.ObjcClassPrefix, "string")+",\n")
	}
	if this.CsharpNamespace != nil {
		s = append(s, "CsharpNamespace: "+valueToGoStringDescriptor(this.CsharpNamespace, "string")+",\n")
	}
	if this.JavananoUseDeprecatedPackage != nil {
		s = append(s, "JavananoUseDeprecatedPackage: "+valueToGoStringDescriptor(this.JavananoUseDeprecatedPackage, "bool")+",\n")
	}
	if this.UninterpretedOption != nil {
		s = append(s, "UninterpretedOption: "+fmt.Sprintf("%#v", this.UninterpretedOption)+",\n")
	}
	if this.XXX_extensions != nil {
		s = append(s, "XXX_extensions: "+extensionToGoStringDescriptor(this.XXX_extensions)+",\n")
	}
	if this.XXX_unrecognized != nil {
		s = append(s, "XXX_unrecognized:"+fmt.Sprintf("%#v", this.XXX_unrecognized)+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func (this *MessageOptions) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 9)
	s = append(s, "&descriptor.MessageOptions{")
	if this.MessageSetWireFormat != nil {
		s = append(s, "MessageSetWireFormat: "+valueToGoStringDescriptor(this.MessageSetWireFormat, "bool")+",\n")
	}
	if this.NoStandardDescriptorAccessor != nil {
		s = append(s, "NoStandardDescriptorAccessor: "+valueToGoStringDescriptor(this.NoStandardDescriptorAccessor, "bool")+",\n")
	}
	if this.Deprecated != nil {
		s = append(s, "Deprecated: "+valueToGoStringDescriptor(this.Deprecated, "bool")+",\n")
	}
	if this.MapEntry != nil {
		s = append(s, "MapEntry: "+valueToGoStringDescriptor(this.MapEntry, "bool")+",\n")
	}
	if this.UninterpretedOption != nil {
		s = append(s, "UninterpretedOption: "+fmt.Sprintf("%#v", this.UninterpretedOption)+",\n")
	}
	if this.XXX_extensions != nil {
		s = append(s, "XXX_extensions: "+extensionToGoStringDescriptor(this.XXX_extensions)+",\n")
	}
	if this.XXX_unrecognized != nil {
		s = append(s, "XXX_unrecognized:"+fmt.Sprintf("%#v", this.XXX_unrecognized)+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func (this *FieldOptions) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 11)
	s = append(s, "&descriptor.FieldOptions{")
	if this.Ctype != nil {
		s = append(s, "Ctype: "+valueToGoStringDescriptor(this.Ctype, "descriptor.FieldOptions_CType")+",\n")
	}
	if this.Packed != nil {
		s = append(s, "Packed: "+valueToGoStringDescriptor(this.Packed, "bool")+",\n")
	}
	if this.Jstype != nil {
		s = append(s, "Jstype: "+valueToGoStringDescriptor(this.Jstype, "descriptor.FieldOptions_JSType")+",\n")
	}
	if this.Lazy != nil {
		s = append(s, "Lazy: "+valueToGoStringDescriptor(this.Lazy, "bool")+",\n")
	}
	if this.Deprecated != nil {
		s = append(s, "Deprecated: "+valueToGoStringDescriptor(this.Deprecated, "bool")+",\n")
	}
	if this.Weak != nil {
		s = append(s, "Weak: "+valueToGoStringDescriptor(this.Weak, "bool")+",\n")
	}
	if this.UninterpretedOption != nil {
		s = append(s, "UninterpretedOption: "+fmt.Sprintf("%#v", this.UninterpretedOption)+",\n")
	}
	if this.XXX_extensions != nil {
		s = append(s, "XXX_extensions: "+extensionToGoStringDescriptor(this.XXX_extensions)+",\n")
	}
	if this.XXX_unrecognized != nil {
		s = append(s, "XXX_unrecognized:"+fmt.Sprintf("%#v", this.XXX_unrecognized)+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func (this *EnumOptions) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 7)
	s = append(s, "&descriptor.EnumOptions{")
	if this.AllowAlias != nil {
		s = append(s, "AllowAlias: "+valueToGoStringDescriptor(this.AllowAlias, "bool")+",\n")
	}
	if this.Deprecated != nil {
		s = append(s, "Deprecated: "+valueToGoStringDescriptor(this.Deprecated, "bool")+",\n")
	}
	if this.UninterpretedOption != nil {
		s = append(s, "UninterpretedOption: "+fmt.Sprintf("%#v", this.UninterpretedOption)+",\n")
	}
	if this.XXX_extensions != nil {
		s = append(s, "XXX_extensions: "+extensionToGoStringDescriptor(this.XXX_extensions)+",\n")
	}
	if this.XXX_unrecognized != nil {
		s = append(s, "XXX_unrecognized:"+fmt.Sprintf("%#v", this.XXX_unrecognized)+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func (this *EnumValueOptions) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 6)
	s = append(s, "&descriptor.EnumValueOptions{")
	if this.Deprecated != nil {
		s = append(s, "Deprecated: "+valueToGoStringDescriptor(this.Deprecated, "bool")+",\n")
	}
	if this.UninterpretedOption != nil {
		s = append(s, "UninterpretedOption: "+fmt.Sprintf("%#v", this.UninterpretedOption)+",\n")
	}
	if this.XXX_extensions != nil {
		s = append(s, "XXX_extensions: "+extensionToGoStringDescriptor(this.XXX_extensions)+",\n")
	}
	if this.XXX_unrecognized != nil {
		s = append(s, "XXX_unrecognized:"+fmt.Sprintf("%#v", this.XXX_unrecognized)+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func (this *ServiceOptions) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 6)
	s = append(s, "&descriptor.ServiceOptions{")
	if this.Deprecated != nil {
		s = append(s, "Deprecated: "+valueToGoStringDescriptor(this.Deprecated, "bool")+",\n")
	}
	if this.UninterpretedOption != nil {
		s = append(s, "UninterpretedOption: "+fmt.Sprintf("%#v", this.UninterpretedOption)+",\n")
	}
	if this.XXX_extensions != nil {
		s = append(s, "XXX_extensions: "+extensionToGoStringDescriptor(this.XXX_extensions)+",\n")
	}
	if this.XXX_unrecognized != nil {
		s = append(s, "XXX_unrecognized:"+fmt.Sprintf("%#v", this.XXX_unrecognized)+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func (this *MethodOptions) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 6)
	s = append(s, "&descriptor.MethodOptions{")
	if this.Deprecated != nil {
		s = append(s, "Deprecated: "+valueToGoStringDescriptor(this.Deprecated, "bool")+",\n")
	}
	if this.UninterpretedOption != nil {
		s = append(s, "UninterpretedOption: "+fmt.Sprintf("%#v", this.UninterpretedOption)+",\n")
	}
	if this.XXX_extensions != nil {
		s = append(s, "XXX_extensions: "+extensionToGoStringDescriptor(this.XXX_extensions)+",\n")
	}
	if this.XXX_unrecognized != nil {
		s = append(s, "XXX_unrecognized:"+fmt.Sprintf("%#v", this.XXX_unrecognized)+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func (this *UninterpretedOption) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 11)
	s = append(s, "&descriptor.UninterpretedOption{")
	if this.Name != nil {
		s = append(s, "Name: "+fmt.Sprintf("%#v", this.Name)+",\n")
	}
	if this.IdentifierValue != nil {
		s = append(s, "IdentifierValue: "+valueToGoStringDescriptor(this.IdentifierValue, "string")+",\n")
	}
	if this.PositiveIntValue != nil {
		s = append(s, "PositiveIntValue: "+valueToGoStringDescriptor(this.PositiveIntValue, "uint64")+",\n")
	}
	if this.NegativeIntValue != nil {
		s = append(s, "NegativeIntValue: "+valueToGoStringDescriptor(this.NegativeIntValue, "int64")+",\n")
	}
	if this.DoubleValue != nil {
		s = append(s, "DoubleValue: "+valueToGoStringDescriptor(this.DoubleValue, "float64")+",\n")
	}
	if this.StringValue != nil {
		s = append(s, "StringValue: "+valueToGoStringDescriptor(this.StringValue, "byte")+",\n")
	}
	if this.AggregateValue != nil {
		s = append(s, "AggregateValue: "+valueToGoStringDescriptor(this.AggregateValue, "string")+",\n")
	}
	if this.XXX_unrecognized != nil {
		s = append(s, "XXX_unrecognized:"+fmt.Sprintf("%#v", this.XXX_unrecognized)+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func (this *UninterpretedOption_NamePart) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 6)
	s = append(s, "&descriptor.UninterpretedOption_NamePart{")
	if this.NamePart != nil {
		s = append(s, "NamePart: "+valueToGoStringDescriptor(this.NamePart, "string")+",\n")
	}
	if this.IsExtension != nil {
		s = append(s, "IsExtension: "+valueToGoStringDescriptor(this.IsExtension, "bool")+",\n")
	}
	if this.XXX_unrecognized != nil {
		s = append(s, "XXX_unrecognized:"+fmt.Sprintf("%#v", this.XXX_unrecognized)+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func (this *SourceCodeInfo) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 5)
	s = append(s, "&descriptor.SourceCodeInfo{")
	if this.Location != nil {
		s = append(s, "Location: "+fmt.Sprintf("%#v", this.Location)+",\n")
	}
	if this.XXX_unrecognized != nil {
		s = append(s, "XXX_unrecognized:"+fmt.Sprintf("%#v", this.XXX_unrecognized)+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func (this *SourceCodeInfo_Location) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 9)
	s = append(s, "&descriptor.SourceCodeInfo_Location{")
	if this.Path != nil {
		s = append(s, "Path: "+fmt.Sprintf("%#v", this.Path)+",\n")
	}
	if this.Span != nil {
		s = append(s, "Span: "+fmt.Sprintf("%#v", this.Span)+",\n")
	}
	if this.LeadingComments != nil {
		s = append(s, "LeadingComments: "+valueToGoStringDescriptor(this.LeadingComments, "string")+",\n")
	}
	if this.TrailingComments != nil {
		s = append(s, "TrailingComments: "+valueToGoStringDescriptor(this.TrailingComments, "string")+",\n")
	}
	if this.LeadingDetachedComments != nil {
		s = append(s, "LeadingDetachedComments: "+fmt.Sprintf("%#v", this.LeadingDetachedComments)+",\n")
	}
	if this.XXX_unrecognized != nil {
		s = append(s, "XXX_unrecognized:"+fmt.Sprintf("%#v", this.XXX_unrecognized)+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringDescriptor(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}
func extensionToGoStringDescriptor(e map[int32]github_com_gogo_protobuf_proto.Extension) string {
	if e == nil {
		return "nil"
	}
	s := "map[int32]proto.Extension{"
	keys := make([]int, 0, len(e))
	for k := range e {
		keys = append(keys, int(k))
	}
	sort.Ints(keys)
	ss := []string{}
	for _, k := range keys {
		ss = append(ss, strconv.Itoa(k)+": "+e[int32(k)].GoString())
	}
	s += strings.Join(ss, ",") + "}"
	return s
}
