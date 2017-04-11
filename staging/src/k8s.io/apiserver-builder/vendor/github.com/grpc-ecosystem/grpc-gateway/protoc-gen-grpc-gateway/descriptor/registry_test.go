package descriptor

import (
	"testing"

	"github.com/golang/protobuf/proto"
	descriptor "github.com/golang/protobuf/protoc-gen-go/descriptor"
	plugin "github.com/golang/protobuf/protoc-gen-go/plugin"
)

func loadFile(t *testing.T, reg *Registry, src string) *descriptor.FileDescriptorProto {
	var file descriptor.FileDescriptorProto
	if err := proto.UnmarshalText(src, &file); err != nil {
		t.Fatalf("proto.UnmarshalText(%s, &file) failed with %v; want success", src, err)
	}
	reg.loadFile(&file)
	return &file
}

func load(t *testing.T, reg *Registry, src string) error {
	var req plugin.CodeGeneratorRequest
	if err := proto.UnmarshalText(src, &req); err != nil {
		t.Fatalf("proto.UnmarshalText(%s, &file) failed with %v; want success", src, err)
	}
	return reg.Load(&req)
}

func TestLoadFile(t *testing.T) {
	reg := NewRegistry()
	fd := loadFile(t, reg, `
		name: 'example.proto'
		package: 'example'
		message_type <
			name: 'ExampleMessage'
			field <
				name: 'str'
				label: LABEL_OPTIONAL
				type: TYPE_STRING
				number: 1
			>
		>
	`)

	file := reg.files["example.proto"]
	if file == nil {
		t.Errorf("reg.files[%q] = nil; want non-nil", "example.proto")
		return
	}
	wantPkg := GoPackage{Path: ".", Name: "example"}
	if got, want := file.GoPkg, wantPkg; got != want {
		t.Errorf("file.GoPkg = %#v; want %#v", got, want)
	}

	msg, err := reg.LookupMsg("", ".example.ExampleMessage")
	if err != nil {
		t.Errorf("reg.LookupMsg(%q, %q)) failed with %v; want success", "", ".example.ExampleMessage", err)
		return
	}
	if got, want := msg.DescriptorProto, fd.MessageType[0]; got != want {
		t.Errorf("reg.lookupMsg(%q, %q).DescriptorProto = %#v; want %#v", "", ".example.ExampleMessage", got, want)
	}
	if got, want := msg.File, file; got != want {
		t.Errorf("msg.File = %v; want %v", got, want)
	}
	if got := msg.Outers; got != nil {
		t.Errorf("msg.Outers = %v; want %v", got, nil)
	}
	if got, want := len(msg.Fields), 1; got != want {
		t.Errorf("len(msg.Fields) = %d; want %d", got, want)
	} else if got, want := msg.Fields[0].FieldDescriptorProto, fd.MessageType[0].Field[0]; got != want {
		t.Errorf("msg.Fields[0].FieldDescriptorProto = %v; want %v", got, want)
	} else if got, want := msg.Fields[0].Message, msg; got != want {
		t.Errorf("msg.Fields[0].Message = %v; want %v", got, want)
	}

	if got, want := len(file.Messages), 1; got != want {
		t.Errorf("file.Meeesages = %#v; want %#v", file.Messages, []*Message{msg})
	}
	if got, want := file.Messages[0], msg; got != want {
		t.Errorf("file.Meeesages[0] = %v; want %v", got, want)
	}
}

func TestLoadFileNestedPackage(t *testing.T) {
	reg := NewRegistry()
	loadFile(t, reg, `
		name: 'example.proto'
		package: 'example.nested.nested2'
	`)

	file := reg.files["example.proto"]
	if file == nil {
		t.Errorf("reg.files[%q] = nil; want non-nil", "example.proto")
		return
	}
	wantPkg := GoPackage{Path: ".", Name: "example_nested_nested2"}
	if got, want := file.GoPkg, wantPkg; got != want {
		t.Errorf("file.GoPkg = %#v; want %#v", got, want)
	}
}

func TestLoadFileWithDir(t *testing.T) {
	reg := NewRegistry()
	loadFile(t, reg, `
		name: 'path/to/example.proto'
		package: 'example'
	`)

	file := reg.files["path/to/example.proto"]
	if file == nil {
		t.Errorf("reg.files[%q] = nil; want non-nil", "example.proto")
		return
	}
	wantPkg := GoPackage{Path: "path/to", Name: "example"}
	if got, want := file.GoPkg, wantPkg; got != want {
		t.Errorf("file.GoPkg = %#v; want %#v", got, want)
	}
}

func TestLoadFileWithoutPackage(t *testing.T) {
	reg := NewRegistry()
	loadFile(t, reg, `
		name: 'path/to/example_file.proto'
	`)

	file := reg.files["path/to/example_file.proto"]
	if file == nil {
		t.Errorf("reg.files[%q] = nil; want non-nil", "example.proto")
		return
	}
	wantPkg := GoPackage{Path: "path/to", Name: "example_file"}
	if got, want := file.GoPkg, wantPkg; got != want {
		t.Errorf("file.GoPkg = %#v; want %#v", got, want)
	}
}

func TestLoadFileWithMapping(t *testing.T) {
	reg := NewRegistry()
	reg.AddPkgMap("path/to/example.proto", "example.com/proj/example/proto")
	loadFile(t, reg, `
		name: 'path/to/example.proto'
		package: 'example'
	`)

	file := reg.files["path/to/example.proto"]
	if file == nil {
		t.Errorf("reg.files[%q] = nil; want non-nil", "example.proto")
		return
	}
	wantPkg := GoPackage{Path: "example.com/proj/example/proto", Name: "example"}
	if got, want := file.GoPkg, wantPkg; got != want {
		t.Errorf("file.GoPkg = %#v; want %#v", got, want)
	}
}

func TestLoadFileWithPackageNameCollision(t *testing.T) {
	reg := NewRegistry()
	loadFile(t, reg, `
		name: 'path/to/another.proto'
		package: 'example'
	`)
	loadFile(t, reg, `
		name: 'path/to/example.proto'
		package: 'example'
	`)
	if err := reg.ReserveGoPackageAlias("ioutil", "io/ioutil"); err != nil {
		t.Fatalf("reg.ReserveGoPackageAlias(%q) failed with %v; want success", "ioutil", err)
	}
	loadFile(t, reg, `
		name: 'path/to/ioutil.proto'
		package: 'ioutil'
	`)

	file := reg.files["path/to/another.proto"]
	if file == nil {
		t.Errorf("reg.files[%q] = nil; want non-nil", "path/to/another.proto")
		return
	}
	wantPkg := GoPackage{Path: "path/to", Name: "example"}
	if got, want := file.GoPkg, wantPkg; got != want {
		t.Errorf("file.GoPkg = %#v; want %#v", got, want)
	}

	file = reg.files["path/to/example.proto"]
	if file == nil {
		t.Errorf("reg.files[%q] = nil; want non-nil", "path/to/example.proto")
		return
	}
	wantPkg = GoPackage{Path: "path/to", Name: "example", Alias: ""}
	if got, want := file.GoPkg, wantPkg; got != want {
		t.Errorf("file.GoPkg = %#v; want %#v", got, want)
	}

	file = reg.files["path/to/ioutil.proto"]
	if file == nil {
		t.Errorf("reg.files[%q] = nil; want non-nil", "path/to/ioutil.proto")
		return
	}
	wantPkg = GoPackage{Path: "path/to", Name: "ioutil", Alias: "ioutil_0"}
	if got, want := file.GoPkg, wantPkg; got != want {
		t.Errorf("file.GoPkg = %#v; want %#v", got, want)
	}
}

func TestLoadFileWithIdenticalGoPkg(t *testing.T) {
	reg := NewRegistry()
	reg.AddPkgMap("path/to/another.proto", "example.com/example")
	reg.AddPkgMap("path/to/example.proto", "example.com/example")
	loadFile(t, reg, `
		name: 'path/to/another.proto'
		package: 'example'
	`)
	loadFile(t, reg, `
		name: 'path/to/example.proto'
		package: 'example'
	`)

	file := reg.files["path/to/example.proto"]
	if file == nil {
		t.Errorf("reg.files[%q] = nil; want non-nil", "example.proto")
		return
	}
	wantPkg := GoPackage{Path: "example.com/example", Name: "example"}
	if got, want := file.GoPkg, wantPkg; got != want {
		t.Errorf("file.GoPkg = %#v; want %#v", got, want)
	}

	file = reg.files["path/to/another.proto"]
	if file == nil {
		t.Errorf("reg.files[%q] = nil; want non-nil", "example.proto")
		return
	}
	wantPkg = GoPackage{Path: "example.com/example", Name: "example"}
	if got, want := file.GoPkg, wantPkg; got != want {
		t.Errorf("file.GoPkg = %#v; want %#v", got, want)
	}
}

func TestLoadFileWithPrefix(t *testing.T) {
	reg := NewRegistry()
	reg.SetPrefix("third_party")
	loadFile(t, reg, `
		name: 'path/to/example.proto'
		package: 'example'
	`)

	file := reg.files["path/to/example.proto"]
	if file == nil {
		t.Errorf("reg.files[%q] = nil; want non-nil", "example.proto")
		return
	}
	wantPkg := GoPackage{Path: "third_party/path/to", Name: "example"}
	if got, want := file.GoPkg, wantPkg; got != want {
		t.Errorf("file.GoPkg = %#v; want %#v", got, want)
	}
}

func TestLookupMsgWithoutPackage(t *testing.T) {
	reg := NewRegistry()
	fd := loadFile(t, reg, `
		name: 'example.proto'
		message_type <
			name: 'ExampleMessage'
			field <
				name: 'str'
				label: LABEL_OPTIONAL
				type: TYPE_STRING
				number: 1
			>
		>
	`)

	msg, err := reg.LookupMsg("", ".ExampleMessage")
	if err != nil {
		t.Errorf("reg.LookupMsg(%q, %q)) failed with %v; want success", "", ".ExampleMessage", err)
		return
	}
	if got, want := msg.DescriptorProto, fd.MessageType[0]; got != want {
		t.Errorf("reg.lookupMsg(%q, %q).DescriptorProto = %#v; want %#v", "", ".ExampleMessage", got, want)
	}
}

func TestLookupMsgWithNestedPackage(t *testing.T) {
	reg := NewRegistry()
	fd := loadFile(t, reg, `
		name: 'example.proto'
		package: 'nested.nested2.mypackage'
		message_type <
			name: 'ExampleMessage'
			field <
				name: 'str'
				label: LABEL_OPTIONAL
				type: TYPE_STRING
				number: 1
			>
		>
	`)

	for _, name := range []string{
		"nested.nested2.mypackage.ExampleMessage",
		"nested2.mypackage.ExampleMessage",
		"mypackage.ExampleMessage",
		"ExampleMessage",
	} {
		msg, err := reg.LookupMsg("nested.nested2.mypackage", name)
		if err != nil {
			t.Errorf("reg.LookupMsg(%q, %q)) failed with %v; want success", ".nested.nested2.mypackage", name, err)
			return
		}
		if got, want := msg.DescriptorProto, fd.MessageType[0]; got != want {
			t.Errorf("reg.lookupMsg(%q, %q).DescriptorProto = %#v; want %#v", ".nested.nested2.mypackage", name, got, want)
		}
	}

	for _, loc := range []string{
		".nested.nested2.mypackage",
		"nested.nested2.mypackage",
		".nested.nested2",
		"nested.nested2",
		".nested",
		"nested",
		".",
		"",
		"somewhere.else",
	} {
		name := "nested.nested2.mypackage.ExampleMessage"
		msg, err := reg.LookupMsg(loc, name)
		if err != nil {
			t.Errorf("reg.LookupMsg(%q, %q)) failed with %v; want success", loc, name, err)
			return
		}
		if got, want := msg.DescriptorProto, fd.MessageType[0]; got != want {
			t.Errorf("reg.lookupMsg(%q, %q).DescriptorProto = %#v; want %#v", loc, name, got, want)
		}
	}

	for _, loc := range []string{
		".nested.nested2.mypackage",
		"nested.nested2.mypackage",
		".nested.nested2",
		"nested.nested2",
		".nested",
		"nested",
	} {
		name := "nested2.mypackage.ExampleMessage"
		msg, err := reg.LookupMsg(loc, name)
		if err != nil {
			t.Errorf("reg.LookupMsg(%q, %q)) failed with %v; want success", loc, name, err)
			return
		}
		if got, want := msg.DescriptorProto, fd.MessageType[0]; got != want {
			t.Errorf("reg.lookupMsg(%q, %q).DescriptorProto = %#v; want %#v", loc, name, got, want)
		}
	}
}

func TestLoadWithInconsistentTargetPackage(t *testing.T) {
	for _, spec := range []struct {
		req        string
		consistent bool
	}{
		// root package, no explicit go package
		{
			req: `
				file_to_generate: 'a.proto'
				file_to_generate: 'b.proto'
				proto_file <
					name: 'a.proto'
					message_type < name: 'A' >
					service <
						name: "AService"
						method <
							name: "Meth"
							input_type: "A"
							output_type: "A"
							options <
								[google.api.http] < post: "/v1/a" body: "*" >
							>
						>
					>
				>
				proto_file <
					name: 'b.proto'
					message_type < name: 'B' >
					service <
						name: "BService"
						method <
							name: "Meth"
							input_type: "B"
							output_type: "B"
							options <
								[google.api.http] < post: "/v1/b" body: "*" >
							>
						>
					>
				>
			`,
			consistent: false,
		},
		// named package, no explicit go package
		{
			req: `
				file_to_generate: 'a.proto'
				file_to_generate: 'b.proto'
				proto_file <
					name: 'a.proto'
					package: 'example.foo'
					message_type < name: 'A' >
					service <
						name: "AService"
						method <
							name: "Meth"
							input_type: "A"
							output_type: "A"
							options <
								[google.api.http] < post: "/v1/a" body: "*" >
							>
						>
					>
				>
				proto_file <
					name: 'b.proto'
					package: 'example.foo'
					message_type < name: 'B' >
					service <
						name: "BService"
						method <
							name: "Meth"
							input_type: "B"
							output_type: "B"
							options <
								[google.api.http] < post: "/v1/b" body: "*" >
							>
						>
					>
				>
			`,
			consistent: true,
		},
		// root package, explicit go package
		{
			req: `
				file_to_generate: 'a.proto'
				file_to_generate: 'b.proto'
				proto_file <
					name: 'a.proto'
					options < go_package: 'foo' >
					message_type < name: 'A' >
					service <
						name: "AService"
						method <
							name: "Meth"
							input_type: "A"
							output_type: "A"
							options <
								[google.api.http] < post: "/v1/a" body: "*" >
							>
						>
					>
				>
				proto_file <
					name: 'b.proto'
					options < go_package: 'foo' >
					message_type < name: 'B' >
					service <
						name: "BService"
						method <
							name: "Meth"
							input_type: "B"
							output_type: "B"
							options <
								[google.api.http] < post: "/v1/b" body: "*" >
							>
						>
					>
				>
			`,
			consistent: true,
		},
		// named package, explicit go package
		{
			req: `
				file_to_generate: 'a.proto'
				file_to_generate: 'b.proto'
				proto_file <
					name: 'a.proto'
					package: 'example.foo'
					options < go_package: 'foo' >
					message_type < name: 'A' >
					service <
						name: "AService"
						method <
							name: "Meth"
							input_type: "A"
							output_type: "A"
							options <
								[google.api.http] < post: "/v1/a" body: "*" >
							>
						>
					>
				>
				proto_file <
					name: 'b.proto'
					package: 'example.foo'
					options < go_package: 'foo' >
					message_type < name: 'B' >
					service <
						name: "BService"
						method <
							name: "Meth"
							input_type: "B"
							output_type: "B"
							options <
								[google.api.http] < post: "/v1/b" body: "*" >
							>
						>
					>
				>
			`,
			consistent: true,
		},
	} {
		reg := NewRegistry()
		err := load(t, reg, spec.req)
		if got, want := err == nil, spec.consistent; got != want {
			if want {
				t.Errorf("reg.Load(%s) failed with %v; want success", spec.req, err)
				continue
			}
			t.Errorf("reg.Load(%s) succeeded; want an package inconsistency error", spec.req)
		}
	}
}
