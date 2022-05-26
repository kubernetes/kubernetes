package unused

//go:generate stringer -type edgeKind
type edgeKind uint64

func (e edgeKind) is(o edgeKind) bool {
	return e&o != 0
}

const (
	edgeAlias edgeKind = 1 << iota
	edgeBlankField
	edgeAnonymousStruct
	edgeCgoExported
	edgeConstGroup
	edgeElementType
	edgeEmbeddedInterface
	edgeExportedConstant
	edgeExportedField
	edgeExportedFunction
	edgeExportedMethod
	edgeExportedType
	edgeExportedVariable
	edgeExtendsExportedFields
	edgeExtendsExportedMethodSet
	edgeFieldAccess
	edgeFunctionArgument
	edgeFunctionResult
	edgeFunctionSignature
	edgeImplements
	edgeInstructionOperand
	edgeInterfaceCall
	edgeInterfaceMethod
	edgeKeyType
	edgeLinkname
	edgeMainFunction
	edgeNamedType
	edgeNetRPCRegister
	edgeNoCopySentinel
	edgeProvidesMethod
	edgeReceiver
	edgeRuntimeFunction
	edgeSignature
	edgeStructConversion
	edgeTestSink
	edgeTupleElement
	edgeType
	edgeTypeName
	edgeUnderlyingType
	edgePointerType
	edgeUnsafeConversion
	edgeUsedConstant
	edgeVarDecl
	edgeIgnored
)
