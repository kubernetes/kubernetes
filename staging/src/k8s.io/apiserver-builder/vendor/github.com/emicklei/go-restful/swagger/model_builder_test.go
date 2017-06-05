package swagger

import (
	"encoding/xml"
	"net"
	"reflect"
	"testing"
	"time"
)

type YesNo bool

func (y YesNo) MarshalJSON() ([]byte, error) {
	if y {
		return []byte("yes"), nil
	}
	return []byte("no"), nil
}

// clear && go test -v -test.run TestRef_Issue190 ...swagger
func TestRef_Issue190(t *testing.T) {
	type User struct {
		items []string
	}
	testJsonFromStruct(t, User{}, `{
  "swagger.User": {
   "id": "swagger.User",
   "required": [
    "items"
   ],
   "properties": {
    "items": {
     "type": "array",
     "items": {
      "type": "string"
     }
    }
   }
  }
 }`)
}

func TestWithoutAdditionalFormat(t *testing.T) {
	type mytime struct {
		time.Time
	}
	type usemytime struct {
		t mytime
	}
	testJsonFromStruct(t, usemytime{}, `{
  "swagger.usemytime": {
   "id": "swagger.usemytime",
   "required": [
    "t"
   ],
   "properties": {
    "t": {
     "type": "string"
    }
   }
  }
 }`)
}

func TestWithAdditionalFormat(t *testing.T) {
	type mytime struct {
		time.Time
	}
	type usemytime struct {
		t mytime
	}
	testJsonFromStructWithConfig(t, usemytime{}, `{
  "swagger.usemytime": {
   "id": "swagger.usemytime",
   "required": [
    "t"
   ],
   "properties": {
    "t": {
     "type": "string",
     "format": "date-time"
    }
   }
  }
 }`, &Config{
		SchemaFormatHandler: func(typeName string) string {
			switch typeName {
			case "swagger.mytime":
				return "date-time"
			}
			return ""
		},
	})
}

// clear && go test -v -test.run TestCustomMarshaller_Issue96 ...swagger
func TestCustomMarshaller_Issue96(t *testing.T) {
	type Vote struct {
		What YesNo
	}
	testJsonFromStruct(t, Vote{}, `{
  "swagger.Vote": {
   "id": "swagger.Vote",
   "required": [
    "What"
   ],
   "properties": {
    "What": {
     "type": "string"
    }
   }
  }
 }`)
}

// clear && go test -v -test.run TestPrimitiveTypes ...swagger
func TestPrimitiveTypes(t *testing.T) {
	type Prims struct {
		f float64
		t time.Time
	}
	testJsonFromStruct(t, Prims{}, `{
  "swagger.Prims": {
   "id": "swagger.Prims",
   "required": [
    "f",
    "t"
   ],
   "properties": {
    "f": {
     "type": "number",
     "format": "double"
    },
    "t": {
     "type": "string",
     "format": "date-time"
    }
   }
  }
 }`)
}

// clear && go test -v -test.run TestPrimitivePtrTypes ...swagger
func TestPrimitivePtrTypes(t *testing.T) {
	type Prims struct {
		f *float64
		t *time.Time
		b *bool
		s *string
		i *int
	}
	testJsonFromStruct(t, Prims{}, `{
  "swagger.Prims": {
   "id": "swagger.Prims",
   "required": [
    "f",
    "t",
    "b",
    "s",
    "i"
   ],
   "properties": {
    "b": {
     "type": "boolean"
    },
    "f": {
     "type": "number",
     "format": "double"
    },
    "i": {
     "type": "integer",
     "format": "int32"
    },
    "s": {
     "type": "string"
    },
    "t": {
     "type": "string",
     "format": "date-time"
    }
   }
  }
 }`)
}

// clear && go test -v -test.run TestS1 ...swagger
func TestS1(t *testing.T) {
	type S1 struct {
		Id string
	}
	testJsonFromStruct(t, S1{}, `{
  "swagger.S1": {
   "id": "swagger.S1",
   "required": [
    "Id"
   ],
   "properties": {
    "Id": {
     "type": "string"
    }
   }
  }
 }`)
}

// clear && go test -v -test.run TestS2 ...swagger
func TestS2(t *testing.T) {
	type S2 struct {
		Ids []string
	}
	testJsonFromStruct(t, S2{}, `{
  "swagger.S2": {
   "id": "swagger.S2",
   "required": [
    "Ids"
   ],
   "properties": {
    "Ids": {
     "type": "array",
     "items": {
      "type": "string"
     }
    }
   }
  }
 }`)
}

// clear && go test -v -test.run TestS3 ...swagger
func TestS3(t *testing.T) {
	type NestedS3 struct {
		Id string
	}
	type S3 struct {
		Nested NestedS3
	}
	testJsonFromStruct(t, S3{}, `{
  "swagger.NestedS3": {
   "id": "swagger.NestedS3",
   "required": [
    "Id"
   ],
   "properties": {
    "Id": {
     "type": "string"
    }
   }
  },
  "swagger.S3": {
   "id": "swagger.S3",
   "required": [
    "Nested"
   ],
   "properties": {
    "Nested": {
     "$ref": "swagger.NestedS3"
    }
   }
  }
 }`)
}

type sample struct {
	id       string `swagger:"required"` // TODO
	items    []item
	rootItem item `json:"root" description:"root desc"`
}

type item struct {
	itemName string `json:"name"`
}

// clear && go test -v -test.run TestSampleToModelAsJson ...swagger
func TestSampleToModelAsJson(t *testing.T) {
	testJsonFromStruct(t, sample{items: []item{}}, `{
  "swagger.item": {
   "id": "swagger.item",
   "required": [
    "name"
   ],
   "properties": {
    "name": {
     "type": "string"
    }
   }
  },
  "swagger.sample": {
   "id": "swagger.sample",
   "required": [
    "id",
    "items",
    "root"
   ],
   "properties": {
    "id": {
     "type": "string"
    },
    "items": {
     "type": "array",
     "items": {
      "$ref": "swagger.item"
     }
    },
    "root": {
     "$ref": "swagger.item",
     "description": "root desc"
    }
   }
  }
 }`)
}

func TestJsonTags(t *testing.T) {
	type X struct {
		A string
		B string `json:"-"`
		C int    `json:",string"`
		D int    `json:","`
	}

	expected := `{
  "swagger.X": {
   "id": "swagger.X",
   "required": [
    "A",
    "C",
    "D"
   ],
   "properties": {
    "A": {
     "type": "string"
    },
    "C": {
     "type": "string"
    },
    "D": {
     "type": "integer",
     "format": "int32"
    }
   }
  }
 }`

	testJsonFromStruct(t, X{}, expected)
}

func TestJsonTagOmitempty(t *testing.T) {
	type X struct {
		A int `json:",omitempty"`
		B int `json:"C,omitempty"`
	}

	expected := `{
  "swagger.X": {
   "id": "swagger.X",
   "properties": {
    "A": {
     "type": "integer",
     "format": "int32"
    },
    "C": {
     "type": "integer",
     "format": "int32"
    }
   }
  }
 }`

	testJsonFromStruct(t, X{}, expected)
}

func TestJsonTagName(t *testing.T) {
	type X struct {
		A string `json:"B"`
	}

	expected := `{
  "swagger.X": {
   "id": "swagger.X",
   "required": [
    "B"
   ],
   "properties": {
    "B": {
     "type": "string"
    }
   }
  }
 }`

	testJsonFromStruct(t, X{}, expected)
}

func TestAnonymousStruct(t *testing.T) {
	type X struct {
		A struct {
			B int
		}
	}

	expected := `{
  "swagger.X": {
   "id": "swagger.X",
   "required": [
    "A"
   ],
   "properties": {
    "A": {
     "$ref": "swagger.X.A"
    }
   }
  },
  "swagger.X.A": {
   "id": "swagger.X.A",
   "required": [
    "B"
   ],
   "properties": {
    "B": {
     "type": "integer",
     "format": "int32"
    }
   }
  }
 }`

	testJsonFromStruct(t, X{}, expected)
}

func TestAnonymousPtrStruct(t *testing.T) {
	type X struct {
		A *struct {
			B int
		}
	}

	expected := `{
  "swagger.X": {
   "id": "swagger.X",
   "required": [
    "A"
   ],
   "properties": {
    "A": {
     "$ref": "swagger.X.A"
    }
   }
  },
  "swagger.X.A": {
   "id": "swagger.X.A",
   "required": [
    "B"
   ],
   "properties": {
    "B": {
     "type": "integer",
     "format": "int32"
    }
   }
  }
 }`

	testJsonFromStruct(t, X{}, expected)
}

func TestAnonymousArrayStruct(t *testing.T) {
	type X struct {
		A []struct {
			B int
		}
	}

	expected := `{
  "swagger.X": {
   "id": "swagger.X",
   "required": [
    "A"
   ],
   "properties": {
    "A": {
     "type": "array",
     "items": {
      "$ref": "swagger.X.A"
     }
    }
   }
  },
  "swagger.X.A": {
   "id": "swagger.X.A",
   "required": [
    "B"
   ],
   "properties": {
    "B": {
     "type": "integer",
     "format": "int32"
    }
   }
  }
 }`

	testJsonFromStruct(t, X{}, expected)
}

func TestAnonymousPtrArrayStruct(t *testing.T) {
	type X struct {
		A *[]struct {
			B int
		}
	}

	expected := `{
  "swagger.X": {
   "id": "swagger.X",
   "required": [
    "A"
   ],
   "properties": {
    "A": {
     "type": "array",
     "items": {
      "$ref": "swagger.X.A"
     }
    }
   }
  },
  "swagger.X.A": {
   "id": "swagger.X.A",
   "required": [
    "B"
   ],
   "properties": {
    "B": {
     "type": "integer",
     "format": "int32"
    }
   }
  }
 }`

	testJsonFromStruct(t, X{}, expected)
}

// go test -v -test.run TestEmbeddedStruct_Issue98 ...swagger
func TestEmbeddedStruct_Issue98(t *testing.T) {
	type Y struct {
		A int
	}
	type X struct {
		Y
	}
	testJsonFromStruct(t, X{}, `{
  "swagger.X": {
   "id": "swagger.X",
   "required": [
    "A"
   ],
   "properties": {
    "A": {
     "type": "integer",
     "format": "int32"
    }
   }
  }
 }`)
}

type Dataset struct {
	Names []string
}

// clear && go test -v -test.run TestIssue85 ...swagger
func TestIssue85(t *testing.T) {
	anon := struct{ Datasets []Dataset }{}
	testJsonFromStruct(t, anon, `{
  "struct { Datasets ||swagger.Dataset }": {
   "id": "struct { Datasets ||swagger.Dataset }",
   "required": [
    "Datasets"
   ],
   "properties": {
    "Datasets": {
     "type": "array",
     "items": {
      "$ref": "swagger.Dataset"
     }
    }
   }
  },
  "swagger.Dataset": {
   "id": "swagger.Dataset",
   "required": [
    "Names"
   ],
   "properties": {
    "Names": {
     "type": "array",
     "items": {
      "type": "string"
     }
    }
   }
  }
 }`)
}

type File struct {
	History     []File
	HistoryPtrs []*File
}

// go test -v -test.run TestRecursiveStructure ...swagger
func TestRecursiveStructure(t *testing.T) {
	testJsonFromStruct(t, File{}, `{
  "swagger.File": {
   "id": "swagger.File",
   "required": [
    "History",
    "HistoryPtrs"
   ],
   "properties": {
    "History": {
     "type": "array",
     "items": {
      "$ref": "swagger.File"
     }
    },
    "HistoryPtrs": {
     "type": "array",
     "items": {
      "$ref": "swagger.File"
     }
    }
   }
  }
 }`)
}

type A1 struct {
	B struct {
		Id      int
		Comment string `json:"comment,omitempty"`
	}
}

// go test -v -test.run TestEmbeddedStructA1 ...swagger
func TestEmbeddedStructA1(t *testing.T) {
	testJsonFromStruct(t, A1{}, `{
  "swagger.A1": {
   "id": "swagger.A1",
   "required": [
    "B"
   ],
   "properties": {
    "B": {
     "$ref": "swagger.A1.B"
    }
   }
  },
  "swagger.A1.B": {
   "id": "swagger.A1.B",
   "required": [
    "Id"
   ],
   "properties": {
    "Id": {
     "type": "integer",
     "format": "int32"
    },
    "comment": {
     "type": "string"
    }
   }
  }
 }`)
}

type A2 struct {
	C
}
type C struct {
	Id      int    `json:"B"`
	Comment string `json:"comment,omitempty"`
	Secure  bool   `json:"secure"`
}

// go test -v -test.run TestEmbeddedStructA2 ...swagger
func TestEmbeddedStructA2(t *testing.T) {
	testJsonFromStruct(t, A2{}, `{
  "swagger.A2": {
   "id": "swagger.A2",
   "required": [
    "B",
    "secure"
   ],
   "properties": {
    "B": {
     "type": "integer",
     "format": "int32"
    },
    "comment": {
     "type": "string"
    },
    "secure": {
     "type": "boolean"
    }
   }
  }
 }`)
}

type A3 struct {
	B D
}

type D struct {
	Id int
}

// clear && go test -v -test.run TestStructA3 ...swagger
func TestStructA3(t *testing.T) {
	testJsonFromStruct(t, A3{}, `{
  "swagger.A3": {
   "id": "swagger.A3",
   "required": [
    "B"
   ],
   "properties": {
    "B": {
     "$ref": "swagger.D"
    }
   }
  },
  "swagger.D": {
   "id": "swagger.D",
   "required": [
    "Id"
   ],
   "properties": {
    "Id": {
     "type": "integer",
     "format": "int32"
    }
   }
  }
 }`)
}

type A4 struct {
	D "json:,inline"
}

// clear && go test -v -test.run TestStructA4 ...swagger
func TestEmbeddedStructA4(t *testing.T) {
	testJsonFromStruct(t, A4{}, `{
  "swagger.A4": {
   "id": "swagger.A4",
   "required": [
    "Id"
   ],
   "properties": {
    "Id": {
     "type": "integer",
     "format": "int32"
    }
   }
  }
 }`)
}

type A5 struct {
	D `json:"d"`
}

// clear && go test -v -test.run TestStructA5 ...swagger
func TestEmbeddedStructA5(t *testing.T) {
	testJsonFromStruct(t, A5{}, `{
  "swagger.A5": {
   "id": "swagger.A5",
   "required": [
    "d"
   ],
   "properties": {
    "d": {
     "$ref": "swagger.D"
    }
   }
  },
  "swagger.D": {
   "id": "swagger.D",
   "required": [
    "Id"
   ],
   "properties": {
    "Id": {
     "type": "integer",
     "format": "int32"
    }
   }
  }
 }`)
}

type D2 struct {
	id int
	D  []D
}

type A6 struct {
	D2 "json:,inline"
}

// clear && go test -v -test.run TestStructA4 ...swagger
func TestEmbeddedStructA6(t *testing.T) {
	testJsonFromStruct(t, A6{}, `{
  "swagger.A6": {
   "id": "swagger.A6",
   "required": [
    "id",
    "D"
   ],
   "properties": {
    "D": {
     "type": "array",
     "items": {
      "$ref": "swagger.D"
     }
    },
    "id": {
     "type": "integer",
     "format": "int32"
    }
   }
  },
  "swagger.D": {
   "id": "swagger.D",
   "required": [
    "Id"
   ],
   "properties": {
    "Id": {
     "type": "integer",
     "format": "int32"
    }
   }
  }
 }`)
}

type ObjectId []byte

type Region struct {
	Id   ObjectId `bson:"_id" json:"id"`
	Name string   `bson:"name" json:"name"`
	Type string   `bson:"type" json:"type"`
}

// clear && go test -v -test.run TestRegion_Issue113 ...swagger
func TestRegion_Issue113(t *testing.T) {
	testJsonFromStruct(t, []Region{}, `{
  "||swagger.Region": {
   "id": "||swagger.Region",
   "properties": {}
  },
  "swagger.Region": {
   "id": "swagger.Region",
   "required": [
    "id",
    "name",
    "type"
   ],
   "properties": {
    "id": {
     "type": "string"
    },
    "name": {
     "type": "string"
    },
    "type": {
     "type": "string"
    }
   }
  }
 }`)
}

// clear && go test -v -test.run TestIssue158 ...swagger
func TestIssue158(t *testing.T) {
	type Address struct {
		Country string `json:"country,omitempty"`
	}

	type Customer struct {
		Name    string  `json:"name"`
		Address Address `json:"address"`
	}
	expected := `{
  "swagger.Address": {
   "id": "swagger.Address",
   "properties": {
    "country": {
     "type": "string"
    }
   }
  },
  "swagger.Customer": {
   "id": "swagger.Customer",
   "required": [
    "name",
    "address"
   ],
   "properties": {
    "address": {
     "$ref": "swagger.Address"
    },
    "name": {
     "type": "string"
    }
   }
  }
 }`
	testJsonFromStruct(t, Customer{}, expected)
}

func TestPointers(t *testing.T) {
	type Vote struct {
		What YesNo
	}
	testJsonFromStruct(t, &Vote{}, `{
  "swagger.Vote": {
   "id": "swagger.Vote",
   "required": [
    "What"
   ],
   "properties": {
    "What": {
     "type": "string"
    }
   }
  }
 }`)
}

func TestSlices(t *testing.T) {
	type Address struct {
		Country string `json:"country,omitempty"`
	}
	expected := `{
  "swagger.Address": {
   "id": "swagger.Address",
   "properties": {
    "country": {
     "type": "string"
    }
   }
  },
  "swagger.Customer": {
   "id": "swagger.Customer",
   "required": [
    "name",
    "addresses"
   ],
   "properties": {
    "addresses": {
     "type": "array",
     "items": {
      "$ref": "swagger.Address"
     }
    },
    "name": {
     "type": "string"
    }
   }
  }
 }`
	// both slices (with pointer value and with type value) should have equal swagger representation
	{
		type Customer struct {
			Name      string    `json:"name"`
			Addresses []Address `json:"addresses"`
		}
		testJsonFromStruct(t, Customer{}, expected)
	}
	{
		type Customer struct {
			Name      string     `json:"name"`
			Addresses []*Address `json:"addresses"`
		}
		testJsonFromStruct(t, Customer{}, expected)
	}

}

type Name struct {
	Value string
}

func (n Name) PostBuildModel(m *Model) *Model {
	m.Description = "titles must be upcase"
	return m
}

type TOC struct {
	Titles []Name
}

type Discography struct {
	Title Name
	TOC
}

// clear && go test -v -test.run TestEmbeddedStructPull204 ...swagger
func TestEmbeddedStructPull204(t *testing.T) {
	b := Discography{}
	testJsonFromStruct(t, b, `
{
  "swagger.Discography": {
   "id": "swagger.Discography",
   "required": [
    "Title",
    "Titles"
   ],
   "properties": {
    "Title": {
     "$ref": "swagger.Name"
    },
    "Titles": {
     "type": "array",
     "items": {
      "$ref": "swagger.Name"
     }
    }
   }
  },
  "swagger.Name": {
   "id": "swagger.Name",
   "required": [
    "Value"
   ],
   "properties": {
    "Value": {
     "type": "string"
    }
   }
  }
 }
`)
}

type AddressWithMethod struct {
	Country  string `json:"country,omitempty"`
	PostCode int    `json:"postcode,omitempty"`
}

func (AddressWithMethod) SwaggerDoc() map[string]string {
	return map[string]string{
		"":         "Address doc",
		"country":  "Country doc",
		"postcode": "PostCode doc",
	}
}

func TestDocInMethodSwaggerDoc(t *testing.T) {
	expected := `{
		  "swagger.AddressWithMethod": {
		   "id": "swagger.AddressWithMethod",
		   "description": "Address doc",
		   "properties": {
		    "country": {
		     "type": "string",
		     "description": "Country doc"
		    },
		    "postcode": {
		     "type": "integer",
		     "format": "int32",
		     "description": "PostCode doc"
		    }
		   }
		  }
		 }`
	testJsonFromStruct(t, AddressWithMethod{}, expected)
}

type RefDesc struct {
	f1 *int64 `description:"desc"`
}

func TestPtrDescription(t *testing.T) {
	b := RefDesc{}
	expected := `{
   "swagger.RefDesc": {
    "id": "swagger.RefDesc",
    "required": [
     "f1"
    ],
    "properties": {
     "f1": {
      "type": "integer",
      "format": "int64",
			"description": "desc"
     }
    }
   }
  }`
	testJsonFromStruct(t, b, expected)
}

type A struct {
	B  `json:",inline"`
	C1 `json:"metadata,omitempty"`
}

type B struct {
	SB string
}

type C1 struct {
	SC string
}

func (A) SwaggerDoc() map[string]string {
	return map[string]string{
		"":         "A struct",
		"B":        "B field", // We should not get anything from this
		"metadata": "C1 field",
	}
}

func (B) SwaggerDoc() map[string]string {
	return map[string]string{
		"":   "B struct",
		"SB": "SB field",
	}
}

func (C1) SwaggerDoc() map[string]string {
	return map[string]string{
		"":   "C1 struct",
		"SC": "SC field",
	}
}

func TestNestedStructDescription(t *testing.T) {
	expected := `
{
  "swagger.A": {
   "id": "swagger.A",
   "description": "A struct",
   "required": [
    "SB"
   ],
   "properties": {
    "SB": {
     "type": "string",
     "description": "SB field"
    },
    "metadata": {
     "$ref": "swagger.C1",
     "description": "C1 field"
    }
   }
  },
  "swagger.C1": {
   "id": "swagger.C1",
   "description": "C1 struct",
   "required": [
    "SC"
   ],
   "properties": {
    "SC": {
     "type": "string",
     "description": "SC field"
    }
   }
  }
 }
`
	testJsonFromStruct(t, A{}, expected)
}

// This tests a primitive with type overrides in the struct tags
type FakeInt int
type E struct {
	Id FakeInt `type:"integer"`
	IP net.IP  `type:"string"`
}

func TestOverridenTypeTagE1(t *testing.T) {
	expected := `
{
  "swagger.E": {
   "id": "swagger.E",
   "required": [
    "Id",
    "IP"
   ],
   "properties": {
    "Id": {
     "type": "integer"
    },
    "IP": {
     "type": "string"
    }
   }
  }
 }
`
	testJsonFromStruct(t, E{}, expected)
}

type XmlNamed struct {
	XMLName xml.Name `xml:"user"`
	Id      string   `json:"id" xml:"id"`
	Name    string   `json:"name" xml:"name"`
}

func TestXmlNameStructs(t *testing.T) {
	expected := `
{
  "swagger.XmlNamed": {
   "id": "swagger.XmlNamed",
   "required": [
    "id",
    "name"
   ],
   "properties": {
    "id": {
     "type": "string"
    },
    "name": {
     "type": "string"
    }
   }
  }
 }
`
	testJsonFromStruct(t, XmlNamed{}, expected)
}

func TestNameCustomization(t *testing.T) {
	expected := `
{
  "swagger.A": {
   "id": "swagger.A",
   "description": "A struct",
   "required": [
    "SB"
   ],
   "properties": {
    "SB": {
     "type": "string",
     "description": "SB field"
    },
    "metadata": {
     "$ref": "new.swagger.SpecialC1",
     "description": "C1 field"
    }
   }
  },
  "new.swagger.SpecialC1": {
   "id": "new.swagger.SpecialC1",
   "description": "C1 struct",
   "required": [
    "SC"
   ],
   "properties": {
    "SC": {
     "type": "string",
     "description": "SC field"
    }
   }
  }
 }`

	testJsonFromStructWithConfig(t, A{}, expected, &Config{
		ModelTypeNameHandler: func(t reflect.Type) (string, bool) {
			if t == reflect.TypeOf(C1{}) {
				return "new.swagger.SpecialC1", true
			}
			return "", false
		},
	})
}
