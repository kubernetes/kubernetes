// +build ignore

package codec

import "reflect"

/*

A strict Non-validating namespace-aware XML 1.0 parser and (en|de)coder.

We are attempting this due to perceived issues with encoding/xml:
  - Complicated. It tried to do too much, and is not as simple to use as json.
  - Due to over-engineering, reflection is over-used AND performance suffers:
    java is 6X faster:http://fabsk.eu/blog/category/informatique/dev/golang/
    even PYTHON performs better: http://outgoing.typepad.com/outgoing/2014/07/exploring-golang.html

codec framework will offer the following benefits
  - VASTLY improved performance (when using reflection-mode or codecgen)
  - simplicity and consistency: with the rest of the supported formats
  - all other benefits of codec framework (streaming, codegeneration, etc)

codec is not a drop-in replacement for encoding/xml.
It is a replacement, based on the simplicity and performance of codec.
Look at it like JAXB for Go.

Challenges:

  - Need to output XML preamble, with all namespaces at the right location in the output.
  - Each "end" block is dynamic, so we need to maintain a context-aware stack
  - How to decide when to use an attribute VS an element
  - How to handle chardata, attr, comment EXPLICITLY.
  - Should it output fragments?
    e.g. encoding a bool should just output true OR false, which is not well-formed XML.

Extend the struct tag. See representative example:
  type X struct {
    ID uint8 codec:"xid|http://ugorji.net/x-namespace id,omitempty,toarray,attr,cdata"
  }

Based on this, we encode
  - fields as elements, BUT encode as attributes if struct tag contains ",attr".
  - text as entity-escaped text, BUT encode as CDATA if struct tag contains ",cdata".

In this mode, we only encode as attribute if ",attr" is found, and only encode as CDATA
if ",cdata" is found in the struct tag.

To handle namespaces:
  - XMLHandle is denoted as being namespace-aware.
    Consequently, we WILL use the ns:name pair to encode and decode if defined, else use the plain name.
  - *Encoder and *Decoder know whether the Handle "prefers" namespaces.
  - add *Encoder.getEncName(*structFieldInfo).
    No one calls *structFieldInfo.indexForEncName directly anymore
  - add *Decoder.getStructFieldInfo(encName string) // encName here is either like abc, or h1:nsabc
    No one accesses .encName anymore except in
  - let encode.go and decode.go use these (for consistency)
  - only problem exists for gen.go, where we create a big switch on encName.
    Now, we also have to add a switch on strings.endsWith(kName, encNsName)
    - gen.go will need to have many more methods, and then double-on the 2 switch loops like:
      switch k {
        case "abc" : x.abc()
        case "def" : x.def()
        default {
          switch {
            case !nsAware: panic(...)
            case strings.endsWith("nsabc"): x.abc()
            default: panic(...)
          }
        }
     }

The structure below accomodates this:

  type typeInfo struct {
    sfi []*structFieldInfo // sorted by encName
    sfins // sorted by namespace
    sfia  // sorted, to have those with attributes at the top. Needed to write XML appropriately.
    sfip  // unsorted
  }
  type structFieldInfo struct {
    encName
    nsEncName
    ns string
    attr bool
    cdata bool
  }

indexForEncName is now an internal helper function that takes a sorted array
(one of ti.sfins or ti.sfi). It is only used by *Encoder.getStructFieldInfo(...)

There will be a separate parser from the builder.
The parser will have a method: next() xmlToken method.

xmlToken has fields:
  - type uint8: 0 | ElementStart | ElementEnd | AttrKey | AttrVal | Text
  - value string
  - ns string

SEE: http://www.xml.com/pub/a/98/10/guide0.html?page=3#ENTDECL

The following are skipped when parsing:
  - External Entities (from external file)
  - Notation Declaration e.g. <!NOTATION GIF87A SYSTEM "GIF">
  - Entity Declarations & References
  - XML Declaration (assume UTF-8)
  - XML Directive i.e. <! ... >
  - Other Declarations: Notation, etc.
  - Comment
  - Processing Instruction
  - schema / DTD for validation:
    We are not a VALIDATING parser. Validation is done elsewhere.
    However, some parts of the DTD internal subset are used (SEE BELOW).
    For Attribute List Declarations e.g.
    <!ATTLIST foo:oldjoke name ID #REQUIRED label CDATA #IMPLIED status ( funny | notfunny ) 'funny' >
    We considered using the ATTLIST to get "default" value, but not to validate the contents. (VETOED)

The following XML features are supported
  - Namespace
  - Element
  - Attribute
  - cdata
  - Unicode escape

The following DTD (when as an internal sub-set) features are supported:
  - Internal Entities e.g.
    <!ELEMENT burns "ugorji is cool" > AND entities for the set: [<>&"']
  - Parameter entities e.g.
    <!ENTITY % personcontent "ugorji is cool"> <!ELEMENT burns (%personcontent;)*>

At decode time, a structure containing the following is kept
  - namespace mapping
  - default attribute values
  - all internal entities (<>&"' and others written in the document)

When decode starts, it parses XML namespace declarations and creates a map in the
xmlDecDriver. While parsing, that map continously gets updated.
The only problem happens when a namespace declaration happens on the node that it defines.
e.g. <hn:name xmlns:hn="http://www.ugorji.net" >
To handle this, each Element must be fully parsed at a time,
even if it amounts to multiple tokens which are returned one at a time on request.

xmlns is a special attribute name.
  - It is used to define namespaces, including the default
  - It is never returned as an AttrKey or AttrVal.
  *We may decide later to allow user to use it e.g. you want to parse the xmlns mappings into a field.*

Number, bool, null, mapKey, etc can all be decoded from any xmlToken.
This accomodates map[int]string for example.

It should be possible to create a schema from the types,
or vice versa (generate types from schema with appropriate tags).
This is however out-of-scope from this parsing project.

We should write all namespace information at the first point that it is referenced in the tree,
and use the mapping for all child nodes and attributes. This means that state is maintained
at a point in the tree. This also means that calls to Decode or MustDecode will reset some state.

When decoding, it is important to keep track of entity references and default attribute values.
It seems these can only be stored in the DTD components. We should honor them when decoding.

Configuration for XMLHandle will look like this:

  XMLHandle
    DefaultNS string
    // Encoding:
    NS map[string]string // ns URI to key, used for encoding
    // Decoding: in case ENTITY declared in external schema or dtd, store info needed here
    Entities map[string]string // map of entity rep to character


During encode, if a namespace mapping is not defined for a namespace found on a struct,
then we create a mapping for it using nsN (where N is 1..1000000, and doesn't conflict
with any other namespace mapping).

Note that different fields in a struct can have different namespaces.
However, all fields will default to the namespace on the _struct field (if defined).

An XML document is a name, a map of attributes and a list of children.
Consequently, we cannot "DecodeNaked" into a map[string]interface{} (for example).
We have to "DecodeNaked" into something that resembles XML data.

To support DecodeNaked (decode into nil interface{}) we have to define some "supporting" types:
    type Name struct { // Prefered. Less allocations due to conversions.
      Local string
      Space string
    }
    type Element struct {
      Name Name
      Attrs map[Name]string
      Children []interface{} // each child is either *Element or string
    }
Only two "supporting" types are exposed for XML: Name and Element.

We considered 'type Name string' where Name is like "Space Local" (space-separated).
We decided against it, because each creation of a name would lead to
double allocation (first convert []byte to string, then concatenate them into a string).
The benefit is that it is faster to read Attrs from a map. But given that Element is a value
object, we want to eschew methods and have public exposed variables.

We also considered the following, where xml types were not value objects, and we used
intelligent accessor methods to extract information and for performance.
*** WE DECIDED AGAINST THIS. ***
    type Attr struct {
      Name Name
      Value string
    }
    // Element is a ValueObject: There are no accessor methods.
    // Make element self-contained.
    type Element struct {
      Name Name
      attrsMap map[string]string // where key is "Space Local"
      attrs []Attr
      childrenT []string
      childrenE []Element
      childrenI []int // each child is a index into T or E.
    }
    func (x *Element) child(i) interface{} // returns string or *Element

Per XML spec and our default handling, white space is insignificant between elements,
specifically between parent-child or siblings. White space occuring alone between start
and end element IS significant. However, if xml:space='preserve', then we 'preserve'
all whitespace. This is more critical when doing a DecodeNaked, but MAY not be as critical
when decoding into a typed value.

**Note: there is no xml: namespace. The xml: attributes were defined before namespaces.**
**So treat them as just "directives" that should be interpreted to mean something**.

On encoding, we don't add any prettifying markup (indenting, etc).

A document or element can only be encoded/decoded from/to a struct. In this mode:
  - struct name maps to element name (or tag-info from _struct field)
  - fields are mapped to child elements or attributes

A map is either encoded as attributes on current element, or as a set of child elements.
Maps are encoded as attributes iff their keys and values are primitives (number, bool, string).

A list is encoded as a set of child elements.

Primitives (number, bool, string) are encoded as an element, attribute or text
depending on the context.

Extensions must encode themselves as a text string.

Encoding is tough, specifically when encoding mappings, because we need to encode
as either attribute or element. To do this, we need to default to encoding as attributes,
and then let Encoder inform the Handle when to start encoding as nodes.
i.e. Encoder does something like:

    h.EncodeMapStart()
    h.Encode(), h.Encode(), ...
    h.EncodeMapNotAttrSignal() // this is not a bool, because it's a signal
    h.Encode(), h.Encode(), ...
    h.EncodeEnd()

Only XMLHandle understands this, and will set itself to start encoding as elements.

This support extends to maps. For example, if a struct field is a map, and it has
the struct tag signifying it should be attr, then all its fields are encoded as attributes.
e.g.

    type X struct {
       M map[string]int `codec:"m,attr"` // encode as attributes
    }

Question:
  - if encoding a map, what if map keys have spaces in them???
    Then they cannot be attributes or child elements. Error.

Misc:

  - For attribute values, normalize by trimming beginning and ending white space,
    and converting every white space sequence to a single space.
  - ATTLIST restrictions are enforced.
    e.g. default value of xml:space, skipping xml:XYZ style attributes, etc.
  - Consider supporting NON-STRICT mode (e.g. to handle HTML parsing).
    Some elements e.g. br, hr, etc need not close and should be auto-closed
    ... (see http://www.w3.org/TR/html4/loose.dtd)
    An expansive set of entities are pre-defined.
  - Have easy way to create a HTML parser:
    add a HTML() method to XMLHandle, that will set Strict=false, specify AutoClose,
    and add HTML Entities to the list.
  - Support validating element/attribute XMLName before writing it.
    Keep this behind a flag, which is set to false by default (for performance).
    type XMLHandle struct {
      CheckName bool
    }

ROADMAP (1 weeks):
  - build encoder (1 day)
  - build decoder (based off xmlParser) (1 day)
  - implement xmlParser (2 days).
    Look at encoding/xml for inspiration.
  - integrate and TEST (1 days)
  - write article and post it (1 day)


*/

// ----------- PARSER  -------------------

type xmlTokenType uint8

const (
	_ xmlTokenType = iota << 1
	xmlTokenElemStart
	xmlTokenElemEnd
	xmlTokenAttrKey
	xmlTokenAttrVal
	xmlTokenText
)

type xmlToken struct {
	Type      xmlTokenType
	Value     string
	Namespace string // blank for AttrVal and Text
}

type xmlParser struct {
	r    decReader
	toks []xmlToken // list of tokens.
	ptr  int        // ptr into the toks slice
	done bool       // nothing else to parse. r now returns EOF.
}

func (x *xmlParser) next() (t *xmlToken) {
	// once x.done, or x.ptr == len(x.toks) == 0, then return nil (to signify finish)
	if !x.done && len(x.toks) == 0 {
		x.nextTag()
	}
	// parses one element at a time (into possible many tokens)
	if x.ptr < len(x.toks) {
		t = &(x.toks[x.ptr])
		x.ptr++
		if x.ptr == len(x.toks) {
			x.ptr = 0
			x.toks = x.toks[:0]
		}
	}
	return
}

// nextTag will parses the next element and fill up toks.
// It set done flag if/once EOF is reached.
func (x *xmlParser) nextTag() {
	// TODO: implement.
}

// ----------- ENCODER -------------------

type xmlEncDriver struct {
	e  *Encoder
	w  encWriter
	h  *XMLHandle
	b  [64]byte // scratch
	bs []byte   // scratch
	// s  jsonStack
	noBuiltInTypes
}

// ----------- DECODER -------------------

type xmlDecDriver struct {
	d    *Decoder
	h    *XMLHandle
	r    decReader // *bytesDecReader decReader
	ct   valueType // container type. one of unset, array or map.
	bstr [8]byte   // scratch used for string \UXXX parsing
	b    [64]byte  // scratch

	// wsSkipped bool // whitespace skipped

	// s jsonStack

	noBuiltInTypes
}

// DecodeNaked will decode into an XMLNode

// XMLName is a value object representing a namespace-aware NAME
type XMLName struct {
	Local string
	Space string
}

// XMLNode represents a "union" of the different types of XML Nodes.
// Only one of fields (Text or *Element) is set.
type XMLNode struct {
	Element *Element
	Text    string
}

// XMLElement is a value object representing an fully-parsed XML element.
type XMLElement struct {
	Name  Name
	Attrs map[XMLName]string
	// Children is a list of child nodes, each being a *XMLElement or string
	Children []XMLNode
}

// ----------- HANDLE  -------------------

type XMLHandle struct {
	BasicHandle
	textEncodingType

	DefaultNS string
	NS        map[string]string // ns URI to key, for encoding
	Entities  map[string]string // entity representation to string, for encoding.
}

func (h *XMLHandle) newEncDriver(e *Encoder) encDriver {
	return &xmlEncDriver{e: e, w: e.w, h: h}
}

func (h *XMLHandle) newDecDriver(d *Decoder) decDriver {
	// d := xmlDecDriver{r: r.(*bytesDecReader), h: h}
	hd := xmlDecDriver{d: d, r: d.r, h: h}
	hd.n.bytes = d.b[:]
	return &hd
}

func (h *XMLHandle) SetInterfaceExt(rt reflect.Type, tag uint64, ext InterfaceExt) (err error) {
	return h.SetExt(rt, tag, &setExtWrapper{i: ext})
}

var _ decDriver = (*xmlDecDriver)(nil)
var _ encDriver = (*xmlEncDriver)(nil)
