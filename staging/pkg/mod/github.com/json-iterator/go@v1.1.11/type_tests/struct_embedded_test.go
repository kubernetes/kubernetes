package test

func init() {
	testCases = append(testCases,
		(*struct {
			EmbeddedFloat64
		})(nil),
		(*struct {
			EmbeddedInt32
		})(nil),
		(*struct {
			F1 float64
			StringMarshaler
			F2 int32
		})(nil),
		(*struct {
			EmbeddedMapStringString
		})(nil),
		(*struct {
			*EmbeddedFloat64
		})(nil),
		(*struct {
			*EmbeddedInt32
		})(nil),
		(*struct {
			*EmbeddedMapStringString
		})(nil),
		(*struct {
			*EmbeddedSliceString
		})(nil),
		(*struct {
			*EmbeddedString
		})(nil),
		(*struct {
			*EmbeddedStruct
		})(nil),
		(*struct {
			EmbeddedSliceString
		})(nil),
		(*struct {
			EmbeddedString
		})(nil),
		(*struct {
			EmbeddedString `json:"othername"`
		})(nil),
		(*struct {
			EmbeddedStruct
		})(nil),
		(*struct {
			F1 float64
			StringTextMarshaler
			F2 int32
		})(nil),
		(*OverlapDifferentLevels)(nil),
		(*IgnoreDeeperLevel)(nil),
		(*SameLevel1BothTagged)(nil),
		(*SameLevel1NoTags)(nil),
		(*SameLevel1Tagged)(nil),
		(*SameLevel2BothTagged)(nil),
		(*SameLevel2NoTags)(nil),
		(*SameLevel2Tagged)(nil),
		(*EmbeddedPtr)(nil),
		(*UnnamedLiteral)(nil),
	)
}

type EmbeddedFloat64 float64
type EmbeddedInt32 int32
type EmbeddedMapStringString map[string]string
type EmbeddedSliceString []string
type EmbeddedString string
type EmbeddedStruct struct {
	String string
	Int    int32
	Float  float64
	Struct struct {
		X string
	}
	Slice []string
	Map   map[string]string
}

type OverlapDifferentLevelsE1 struct {
	F1 int32
}

type OverlapDifferentLevelsE2 struct {
	F2 string
}

type OverlapDifferentLevels struct {
	OverlapDifferentLevelsE1
	OverlapDifferentLevelsE2
	F1 string
}

type IgnoreDeeperLevelDoubleEmbedded struct {
	F1 int32 `json:"F1"`
}

type IgnoreDeeperLevelE1 struct {
	IgnoreDeeperLevelDoubleEmbedded
	F1 int32
}

type IgnoreDeeperLevelE2 struct {
	F1 int32 `json:"F1"`
	IgnoreDeeperLevelDoubleEmbedded
}

type IgnoreDeeperLevel struct {
	IgnoreDeeperLevelE1
	IgnoreDeeperLevelE2
}

type SameLevel1BothTaggedE1 struct {
	F1 int32 `json:"F1"`
}

type SameLevel1BothTaggedE2 struct {
	F1 int32 `json:"F1"`
}

type SameLevel1BothTagged struct {
	SameLevel1BothTaggedE1
	SameLevel1BothTaggedE2
}

type SameLevel1NoTagsE1 struct {
	F1 int32
}

type SameLevel1NoTagsE2 struct {
	F1 int32
}

type SameLevel1NoTags struct {
	SameLevel1NoTagsE1
	SameLevel1NoTagsE2
}

type SameLevel1TaggedE1 struct {
	F1 int32
}

type SameLevel1TaggedE2 struct {
	F1 int32 `json:"F1"`
}

type SameLevel1Tagged struct {
	SameLevel1TaggedE1
	SameLevel1TaggedE2
}

type SameLevel2BothTaggedDE1 struct {
	F1 int32 `json:"F1"`
}

type SameLevel2BothTaggedE1 struct {
	SameLevel2BothTaggedDE1
}

// DoubleEmbedded2 TEST ONLY
type SameLevel2BothTaggedDE2 struct {
	F1 int32 `json:"F1"`
}

// Embedded2 TEST ONLY
type SameLevel2BothTaggedE2 struct {
	SameLevel2BothTaggedDE2
}

type SameLevel2BothTagged struct {
	SameLevel2BothTaggedE1
	SameLevel2BothTaggedE2
}

type SameLevel2NoTagsDE1 struct {
	F1 int32
}

type SameLevel2NoTagsE1 struct {
	SameLevel2NoTagsDE1
}

type SameLevel2NoTagsDE2 struct {
	F1 int32
}

type SameLevel2NoTagsE2 struct {
	SameLevel2NoTagsDE2
}

type SameLevel2NoTags struct {
	SameLevel2NoTagsE1
	SameLevel2NoTagsE2
}

// DoubleEmbedded1 TEST ONLY
type SameLevel2TaggedDE1 struct {
	F1 int32
}

// Embedded1 TEST ONLY
type SameLevel2TaggedE1 struct {
	SameLevel2TaggedDE1
}

// DoubleEmbedded2 TEST ONLY
type SameLevel2TaggedDE2 struct {
	F1 int32 `json:"F1"`
}

// Embedded2 TEST ONLY
type SameLevel2TaggedE2 struct {
	SameLevel2TaggedDE2
}

type SameLevel2Tagged struct {
	SameLevel2TaggedE1
	SameLevel2TaggedE2
}

type EmbeddedPtrO1 struct {
	O1F string
}

type EmbeddedPtrOption struct {
	O1 *EmbeddedPtrO1
}

type EmbeddedPtr struct {
	EmbeddedPtrOption `json:","`
}

type UnnamedLiteral struct {
	_ struct{}
}
