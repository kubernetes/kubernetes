// Code generated from /Users/tswadell/go/src/github.com/google/cel-go/parser/gen/CEL.g4 by ANTLR 4.12.0. DO NOT EDIT.

package gen // CEL
import "github.com/antlr/antlr4/runtime/Go/antlr/v4"

// CELListener is a complete listener for a parse tree produced by CELParser.
type CELListener interface {
	antlr.ParseTreeListener

	// EnterStart is called when entering the start production.
	EnterStart(c *StartContext)

	// EnterExpr is called when entering the expr production.
	EnterExpr(c *ExprContext)

	// EnterConditionalOr is called when entering the conditionalOr production.
	EnterConditionalOr(c *ConditionalOrContext)

	// EnterConditionalAnd is called when entering the conditionalAnd production.
	EnterConditionalAnd(c *ConditionalAndContext)

	// EnterRelation is called when entering the relation production.
	EnterRelation(c *RelationContext)

	// EnterCalc is called when entering the calc production.
	EnterCalc(c *CalcContext)

	// EnterMemberExpr is called when entering the MemberExpr production.
	EnterMemberExpr(c *MemberExprContext)

	// EnterLogicalNot is called when entering the LogicalNot production.
	EnterLogicalNot(c *LogicalNotContext)

	// EnterNegate is called when entering the Negate production.
	EnterNegate(c *NegateContext)

	// EnterMemberCall is called when entering the MemberCall production.
	EnterMemberCall(c *MemberCallContext)

	// EnterSelect is called when entering the Select production.
	EnterSelect(c *SelectContext)

	// EnterPrimaryExpr is called when entering the PrimaryExpr production.
	EnterPrimaryExpr(c *PrimaryExprContext)

	// EnterIndex is called when entering the Index production.
	EnterIndex(c *IndexContext)

	// EnterIdentOrGlobalCall is called when entering the IdentOrGlobalCall production.
	EnterIdentOrGlobalCall(c *IdentOrGlobalCallContext)

	// EnterNested is called when entering the Nested production.
	EnterNested(c *NestedContext)

	// EnterCreateList is called when entering the CreateList production.
	EnterCreateList(c *CreateListContext)

	// EnterCreateStruct is called when entering the CreateStruct production.
	EnterCreateStruct(c *CreateStructContext)

	// EnterCreateMessage is called when entering the CreateMessage production.
	EnterCreateMessage(c *CreateMessageContext)

	// EnterConstantLiteral is called when entering the ConstantLiteral production.
	EnterConstantLiteral(c *ConstantLiteralContext)

	// EnterExprList is called when entering the exprList production.
	EnterExprList(c *ExprListContext)

	// EnterListInit is called when entering the listInit production.
	EnterListInit(c *ListInitContext)

	// EnterFieldInitializerList is called when entering the fieldInitializerList production.
	EnterFieldInitializerList(c *FieldInitializerListContext)

	// EnterOptField is called when entering the optField production.
	EnterOptField(c *OptFieldContext)

	// EnterMapInitializerList is called when entering the mapInitializerList production.
	EnterMapInitializerList(c *MapInitializerListContext)

	// EnterOptExpr is called when entering the optExpr production.
	EnterOptExpr(c *OptExprContext)

	// EnterInt is called when entering the Int production.
	EnterInt(c *IntContext)

	// EnterUint is called when entering the Uint production.
	EnterUint(c *UintContext)

	// EnterDouble is called when entering the Double production.
	EnterDouble(c *DoubleContext)

	// EnterString is called when entering the String production.
	EnterString(c *StringContext)

	// EnterBytes is called when entering the Bytes production.
	EnterBytes(c *BytesContext)

	// EnterBoolTrue is called when entering the BoolTrue production.
	EnterBoolTrue(c *BoolTrueContext)

	// EnterBoolFalse is called when entering the BoolFalse production.
	EnterBoolFalse(c *BoolFalseContext)

	// EnterNull is called when entering the Null production.
	EnterNull(c *NullContext)

	// ExitStart is called when exiting the start production.
	ExitStart(c *StartContext)

	// ExitExpr is called when exiting the expr production.
	ExitExpr(c *ExprContext)

	// ExitConditionalOr is called when exiting the conditionalOr production.
	ExitConditionalOr(c *ConditionalOrContext)

	// ExitConditionalAnd is called when exiting the conditionalAnd production.
	ExitConditionalAnd(c *ConditionalAndContext)

	// ExitRelation is called when exiting the relation production.
	ExitRelation(c *RelationContext)

	// ExitCalc is called when exiting the calc production.
	ExitCalc(c *CalcContext)

	// ExitMemberExpr is called when exiting the MemberExpr production.
	ExitMemberExpr(c *MemberExprContext)

	// ExitLogicalNot is called when exiting the LogicalNot production.
	ExitLogicalNot(c *LogicalNotContext)

	// ExitNegate is called when exiting the Negate production.
	ExitNegate(c *NegateContext)

	// ExitMemberCall is called when exiting the MemberCall production.
	ExitMemberCall(c *MemberCallContext)

	// ExitSelect is called when exiting the Select production.
	ExitSelect(c *SelectContext)

	// ExitPrimaryExpr is called when exiting the PrimaryExpr production.
	ExitPrimaryExpr(c *PrimaryExprContext)

	// ExitIndex is called when exiting the Index production.
	ExitIndex(c *IndexContext)

	// ExitIdentOrGlobalCall is called when exiting the IdentOrGlobalCall production.
	ExitIdentOrGlobalCall(c *IdentOrGlobalCallContext)

	// ExitNested is called when exiting the Nested production.
	ExitNested(c *NestedContext)

	// ExitCreateList is called when exiting the CreateList production.
	ExitCreateList(c *CreateListContext)

	// ExitCreateStruct is called when exiting the CreateStruct production.
	ExitCreateStruct(c *CreateStructContext)

	// ExitCreateMessage is called when exiting the CreateMessage production.
	ExitCreateMessage(c *CreateMessageContext)

	// ExitConstantLiteral is called when exiting the ConstantLiteral production.
	ExitConstantLiteral(c *ConstantLiteralContext)

	// ExitExprList is called when exiting the exprList production.
	ExitExprList(c *ExprListContext)

	// ExitListInit is called when exiting the listInit production.
	ExitListInit(c *ListInitContext)

	// ExitFieldInitializerList is called when exiting the fieldInitializerList production.
	ExitFieldInitializerList(c *FieldInitializerListContext)

	// ExitOptField is called when exiting the optField production.
	ExitOptField(c *OptFieldContext)

	// ExitMapInitializerList is called when exiting the mapInitializerList production.
	ExitMapInitializerList(c *MapInitializerListContext)

	// ExitOptExpr is called when exiting the optExpr production.
	ExitOptExpr(c *OptExprContext)

	// ExitInt is called when exiting the Int production.
	ExitInt(c *IntContext)

	// ExitUint is called when exiting the Uint production.
	ExitUint(c *UintContext)

	// ExitDouble is called when exiting the Double production.
	ExitDouble(c *DoubleContext)

	// ExitString is called when exiting the String production.
	ExitString(c *StringContext)

	// ExitBytes is called when exiting the Bytes production.
	ExitBytes(c *BytesContext)

	// ExitBoolTrue is called when exiting the BoolTrue production.
	ExitBoolTrue(c *BoolTrueContext)

	// ExitBoolFalse is called when exiting the BoolFalse production.
	ExitBoolFalse(c *BoolFalseContext)

	// ExitNull is called when exiting the Null production.
	ExitNull(c *NullContext)
}
