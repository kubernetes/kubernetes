// Code generated "gen_filter_op.go"; DO NOT EDIT.

package ir

const (
	FilterInvalidOp FilterOp = 0

	// !$Args[0]
	FilterNotOp FilterOp = 1

	// $Args[0] && $Args[1]
	FilterAndOp FilterOp = 2

	// $Args[0] || $Args[1]
	FilterOrOp FilterOp = 3

	// $Args[0] == $Args[1]
	FilterEqOp FilterOp = 4

	// $Args[0] != $Args[1]
	FilterNeqOp FilterOp = 5

	// $Args[0] > $Args[1]
	FilterGtOp FilterOp = 6

	// $Args[0] < $Args[1]
	FilterLtOp FilterOp = 7

	// $Args[0] >= $Args[1]
	FilterGtEqOp FilterOp = 8

	// $Args[0] <= $Args[1]
	FilterLtEqOp FilterOp = 9

	// m[$Value].Addressable
	// $Value type: string
	FilterVarAddressableOp FilterOp = 10

	// m[$Value].Pure
	// $Value type: string
	FilterVarPureOp FilterOp = 11

	// m[$Value].Const
	// $Value type: string
	FilterVarConstOp FilterOp = 12

	// m[$Value].ConstSlice
	// $Value type: string
	FilterVarConstSliceOp FilterOp = 13

	// m[$Value].Text
	// $Value type: string
	FilterVarTextOp FilterOp = 14

	// m[$Value].Line
	// $Value type: string
	FilterVarLineOp FilterOp = 15

	// m[$Value].Value.Int()
	// $Value type: string
	FilterVarValueIntOp FilterOp = 16

	// m[$Value].Type.Size
	// $Value type: string
	FilterVarTypeSizeOp FilterOp = 17

	// m[$Value].Type.HasPointers()
	// $Value type: string
	FilterVarTypeHasPointersOp FilterOp = 18

	// m[$Value].Filter($Args[0])
	// $Value type: string
	FilterVarFilterOp FilterOp = 19

	// m[$Value].Node.Is($Args[0])
	// $Value type: string
	FilterVarNodeIsOp FilterOp = 20

	// m[$Value].Object.Is($Args[0])
	// $Value type: string
	FilterVarObjectIsOp FilterOp = 21

	// m[$Value].Type.Is($Args[0])
	// $Value type: string
	FilterVarTypeIsOp FilterOp = 22

	// m[$Value].Type.Underlying().Is($Args[0])
	// $Value type: string
	FilterVarTypeUnderlyingIsOp FilterOp = 23

	// m[$Value].Type.OfKind($Args[0])
	// $Value type: string
	FilterVarTypeOfKindOp FilterOp = 24

	// m[$Value].Type.Underlying().OfKind($Args[0])
	// $Value type: string
	FilterVarTypeUnderlyingOfKindOp FilterOp = 25

	// m[$Value].Type.ConvertibleTo($Args[0])
	// $Value type: string
	FilterVarTypeConvertibleToOp FilterOp = 26

	// m[$Value].Type.AssignableTo($Args[0])
	// $Value type: string
	FilterVarTypeAssignableToOp FilterOp = 27

	// m[$Value].Type.Implements($Args[0])
	// $Value type: string
	FilterVarTypeImplementsOp FilterOp = 28

	// m[$Value].Text.Matches($Args[0])
	// $Value type: string
	FilterVarTextMatchesOp FilterOp = 29

	// m.Deadcode()
	FilterDeadcodeOp FilterOp = 30

	// m.GoVersion().Eq($Value)
	// $Value type: string
	FilterGoVersionEqOp FilterOp = 31

	// m.GoVersion().LessThan($Value)
	// $Value type: string
	FilterGoVersionLessThanOp FilterOp = 32

	// m.GoVersion().GreaterThan($Value)
	// $Value type: string
	FilterGoVersionGreaterThanOp FilterOp = 33

	// m.GoVersion().LessEqThan($Value)
	// $Value type: string
	FilterGoVersionLessEqThanOp FilterOp = 34

	// m.GoVersion().GreaterEqThan($Value)
	// $Value type: string
	FilterGoVersionGreaterEqThanOp FilterOp = 35

	// m.File.Imports($Value)
	// $Value type: string
	FilterFileImportsOp FilterOp = 36

	// m.File.PkgPath.Matches($Value)
	// $Value type: string
	FilterFilePkgPathMatchesOp FilterOp = 37

	// m.File.Name.Matches($Value)
	// $Value type: string
	FilterFileNameMatchesOp FilterOp = 38

	// $Value holds a function name
	// $Value type: string
	FilterFilterFuncRefOp FilterOp = 39

	// $Value holds a string constant
	// $Value type: string
	FilterStringOp FilterOp = 40

	// $Value holds an int64 constant
	// $Value type: int64
	FilterIntOp FilterOp = 41

	// m[`$$`].Node.Parent().Is($Args[0])
	FilterRootNodeParentIsOp FilterOp = 42
)

var filterOpNames = map[FilterOp]string{
	FilterInvalidOp:                 `Invalid`,
	FilterNotOp:                     `Not`,
	FilterAndOp:                     `And`,
	FilterOrOp:                      `Or`,
	FilterEqOp:                      `Eq`,
	FilterNeqOp:                     `Neq`,
	FilterGtOp:                      `Gt`,
	FilterLtOp:                      `Lt`,
	FilterGtEqOp:                    `GtEq`,
	FilterLtEqOp:                    `LtEq`,
	FilterVarAddressableOp:          `VarAddressable`,
	FilterVarPureOp:                 `VarPure`,
	FilterVarConstOp:                `VarConst`,
	FilterVarConstSliceOp:           `VarConstSlice`,
	FilterVarTextOp:                 `VarText`,
	FilterVarLineOp:                 `VarLine`,
	FilterVarValueIntOp:             `VarValueInt`,
	FilterVarTypeSizeOp:             `VarTypeSize`,
	FilterVarTypeHasPointersOp:      `VarTypeHasPointers`,
	FilterVarFilterOp:               `VarFilter`,
	FilterVarNodeIsOp:               `VarNodeIs`,
	FilterVarObjectIsOp:             `VarObjectIs`,
	FilterVarTypeIsOp:               `VarTypeIs`,
	FilterVarTypeUnderlyingIsOp:     `VarTypeUnderlyingIs`,
	FilterVarTypeOfKindOp:           `VarTypeOfKind`,
	FilterVarTypeUnderlyingOfKindOp: `VarTypeUnderlyingOfKind`,
	FilterVarTypeConvertibleToOp:    `VarTypeConvertibleTo`,
	FilterVarTypeAssignableToOp:     `VarTypeAssignableTo`,
	FilterVarTypeImplementsOp:       `VarTypeImplements`,
	FilterVarTextMatchesOp:          `VarTextMatches`,
	FilterDeadcodeOp:                `Deadcode`,
	FilterGoVersionEqOp:             `GoVersionEq`,
	FilterGoVersionLessThanOp:       `GoVersionLessThan`,
	FilterGoVersionGreaterThanOp:    `GoVersionGreaterThan`,
	FilterGoVersionLessEqThanOp:     `GoVersionLessEqThan`,
	FilterGoVersionGreaterEqThanOp:  `GoVersionGreaterEqThan`,
	FilterFileImportsOp:             `FileImports`,
	FilterFilePkgPathMatchesOp:      `FilePkgPathMatches`,
	FilterFileNameMatchesOp:         `FileNameMatches`,
	FilterFilterFuncRefOp:           `FilterFuncRef`,
	FilterStringOp:                  `String`,
	FilterIntOp:                     `Int`,
	FilterRootNodeParentIsOp:        `RootNodeParentIs`,
}
var filterOpFlags = map[FilterOp]uint64{
	FilterAndOp:                     flagIsBinaryExpr,
	FilterOrOp:                      flagIsBinaryExpr,
	FilterEqOp:                      flagIsBinaryExpr,
	FilterNeqOp:                     flagIsBinaryExpr,
	FilterGtOp:                      flagIsBinaryExpr,
	FilterLtOp:                      flagIsBinaryExpr,
	FilterGtEqOp:                    flagIsBinaryExpr,
	FilterLtEqOp:                    flagIsBinaryExpr,
	FilterVarAddressableOp:          flagHasVar,
	FilterVarPureOp:                 flagHasVar,
	FilterVarConstOp:                flagHasVar,
	FilterVarConstSliceOp:           flagHasVar,
	FilterVarTextOp:                 flagHasVar,
	FilterVarLineOp:                 flagHasVar,
	FilterVarValueIntOp:             flagHasVar,
	FilterVarTypeSizeOp:             flagHasVar,
	FilterVarTypeHasPointersOp:      flagHasVar,
	FilterVarFilterOp:               flagHasVar,
	FilterVarNodeIsOp:               flagHasVar,
	FilterVarObjectIsOp:             flagHasVar,
	FilterVarTypeIsOp:               flagHasVar,
	FilterVarTypeUnderlyingIsOp:     flagHasVar,
	FilterVarTypeOfKindOp:           flagHasVar,
	FilterVarTypeUnderlyingOfKindOp: flagHasVar,
	FilterVarTypeConvertibleToOp:    flagHasVar,
	FilterVarTypeAssignableToOp:     flagHasVar,
	FilterVarTypeImplementsOp:       flagHasVar,
	FilterVarTextMatchesOp:          flagHasVar,
	FilterStringOp:                  flagIsBasicLit,
	FilterIntOp:                     flagIsBasicLit,
}
