package yaml

import (
	"bytes"
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/util/diff"
)

func TestMarshal(t *testing.T) {
	someString := "foo"
	aStruct := newSomeStruct()
	config := MyConfig{
		ToBeEmbedded: newToBeEmbedded(),
		AStruct:      newSomeStruct(),
		IgnoredField: "ignored",
		AStringSlice: []string{"1", "2"},
		AStructSlice: []someStruct{newSomeStruct(), newSomeStruct()},
		AStructMap: map[string]someStruct{
			"key1": newSomeStruct(),
			"key2": newSomeStruct(),
		},
		EmptyString:          "",
		PointerToEmptyString: nil,
		PointerToString:      &someString,
		PointerToNullStruct:  nil,
		PointerToStruct:      &aStruct,
		AString:              "foo",
		AnInt:                100,
		NoComment:            "noCommentHere",
	}
	b := new(bytes.Buffer)
	err := Marshal(config, b)
	require.NoError(t, err)

	const expected = `# c12.1 c12.2
embeddedString: hi
# c13.1 c13.2
embeddedStruct:
  # c15.1 c15.2
  aString: hello
  # c16.1 c16.2
  aStringSlice:
    - a
    - b
  # c18.1 c18.2
  aStructMap:
    key1:
      # c19.1 c19.2
      anInt: 3
    key2:
      # c19.1 c19.2
      anInt: 3
  # c17.1 c17.2
  aStructSlice:
    -
      # c19.1 c19.2
      anInt: 3
    -
      # c19.1 c19.2
      anInt: 3
  # c14.1 c14.2
  nested:
    # c19.1 c19.2
    anInt: 3
# c20.1 c20.2
aString: foo
# c4.1 c4.2
aStringSlice:
  - 1
  - 2
# c2.1 c2.2
aStruct:
  # c15.1 c15.2
  aString: hello
  # c16.1 c16.2
  aStringSlice:
    - a
    - b
  # c18.1 c18.2
  aStructMap:
    key1:
      # c19.1 c19.2
      anInt: 3
    key2:
      # c19.1 c19.2
      anInt: 3
  # c17.1 c17.2
  aStructSlice:
    -
      # c19.1 c19.2
      anInt: 3
    -
      # c19.1 c19.2
      anInt: 3
  # c14.1 c14.2
  nested:
    # c19.1 c19.2
    anInt: 3
# c6.1 c6.2
aStructMap:
  key1:
    # c15.1 c15.2
    aString: hello
    # c16.1 c16.2
    aStringSlice:
      - a
      - b
    # c18.1 c18.2
    aStructMap:
      key1:
        # c19.1 c19.2
        anInt: 3
      key2:
        # c19.1 c19.2
        anInt: 3
    # c17.1 c17.2
    aStructSlice:
      -
        # c19.1 c19.2
        anInt: 3
      -
        # c19.1 c19.2
        anInt: 3
    # c14.1 c14.2
    nested:
      # c19.1 c19.2
      anInt: 3
  key2:
    # c15.1 c15.2
    aString: hello
    # c16.1 c16.2
    aStringSlice:
      - a
      - b
    # c18.1 c18.2
    aStructMap:
      key1:
        # c19.1 c19.2
        anInt: 3
      key2:
        # c19.1 c19.2
        anInt: 3
    # c17.1 c17.2
    aStructSlice:
      -
        # c19.1 c19.2
        anInt: 3
      -
        # c19.1 c19.2
        anInt: 3
    # c14.1 c14.2
    nested:
      # c19.1 c19.2
      anInt: 3
# c5.1 c5.2
aStructSlice:
  -
    # c15.1 c15.2
    aString: hello
    # c16.1 c16.2
    aStringSlice:
      - a
      - b
    # c18.1 c18.2
    aStructMap:
      key1:
        # c19.1 c19.2
        anInt: 3
      key2:
        # c19.1 c19.2
        anInt: 3
    # c17.1 c17.2
    aStructSlice:
      -
        # c19.1 c19.2
        anInt: 3
      -
        # c19.1 c19.2
        anInt: 3
    # c14.1 c14.2
    nested:
      # c19.1 c19.2
      anInt: 3
  -
    # c15.1 c15.2
    aString: hello
    # c16.1 c16.2
    aStringSlice:
      - a
      - b
    # c18.1 c18.2
    aStructMap:
      key1:
        # c19.1 c19.2
        anInt: 3
      key2:
        # c19.1 c19.2
        anInt: 3
    # c17.1 c17.2
    aStructSlice:
      -
        # c19.1 c19.2
        anInt: 3
      -
        # c19.1 c19.2
        anInt: 3
    # c14.1 c14.2
    nested:
      # c19.1 c19.2
      anInt: 3
# c21.1 c21.2
anInt: 100
# c7.1 c7.2
emptyString: ""
noComment: noCommentHere
# c8.1 c8.2
pointerToEmptyString: null
# c10.1 c10.2
pointerToNullStruct: null
# c9.1 c9.2
pointerToString: foo
# c11.1 c11.2
pointerToStruct:
  # c15.1 c15.2
  aString: hello
  # c16.1 c16.2
  aStringSlice:
    - a
    - b
  # c18.1 c18.2
  aStructMap:
    key1:
      # c19.1 c19.2
      anInt: 3
    key2:
      # c19.1 c19.2
      anInt: 3
  # c17.1 c17.2
  aStructSlice:
    -
      # c19.1 c19.2
      anInt: 3
    -
      # c19.1 c19.2
      anInt: 3
  # c14.1 c14.2
  nested:
    # c19.1 c19.2
    anInt: 3
`
	if e, a := expected, b.String(); e != a {
		t.Errorf("mismatch: %s", diff.StringDiff(e, a))
	}
}

func newToBeEmbedded() ToBeEmbedded {
	return ToBeEmbedded{
		EmbeddedString: "hi",
		EmbeddedStruct: newSomeStruct(),
	}
}

func newSomeStruct() someStruct {
	return someStruct{
		Nested:       newSomeOtherStruct(),
		AString:      "hello",
		AStringSlice: []string{"a", "b"},
		AStructSlice: []someOtherStruct{newSomeOtherStruct(), newSomeOtherStruct()},
		AStructMap: map[string]someOtherStruct{
			"key1": newSomeOtherStruct(),
			"key2": newSomeOtherStruct(),
		},
	}
}

func newSomeOtherStruct() someOtherStruct {
	return someOtherStruct{
		AnInt: 3,
	}
}
