// Code generated from /usr/local/google/home/jdtatum/github/cel-go/parser/gen/CEL.g4 by ANTLR 4.13.1. DO NOT EDIT.

package gen // CEL
import "github.com/antlr4-go/antlr/v4"

// A complete Visitor for a parse tree produced by CELParser.
type CELVisitor interface {
	antlr.ParseTreeVisitor

	// Visit a parse tree produced by CELParser#start.
	VisitStart(ctx *StartContext) interface{}

	// Visit a parse tree produced by CELParser#expr.
	VisitExpr(ctx *ExprContext) interface{}

	// Visit a parse tree produced by CELParser#conditionalOr.
	VisitConditionalOr(ctx *ConditionalOrContext) interface{}

	// Visit a parse tree produced by CELParser#conditionalAnd.
	VisitConditionalAnd(ctx *ConditionalAndContext) interface{}

	// Visit a parse tree produced by CELParser#relation.
	VisitRelation(ctx *RelationContext) interface{}

	// Visit a parse tree produced by CELParser#calc.
	VisitCalc(ctx *CalcContext) interface{}

	// Visit a parse tree produced by CELParser#MemberExpr.
	VisitMemberExpr(ctx *MemberExprContext) interface{}

	// Visit a parse tree produced by CELParser#LogicalNot.
	VisitLogicalNot(ctx *LogicalNotContext) interface{}

	// Visit a parse tree produced by CELParser#Negate.
	VisitNegate(ctx *NegateContext) interface{}

	// Visit a parse tree produced by CELParser#MemberCall.
	VisitMemberCall(ctx *MemberCallContext) interface{}

	// Visit a parse tree produced by CELParser#Select.
	VisitSelect(ctx *SelectContext) interface{}

	// Visit a parse tree produced by CELParser#PrimaryExpr.
	VisitPrimaryExpr(ctx *PrimaryExprContext) interface{}

	// Visit a parse tree produced by CELParser#Index.
	VisitIndex(ctx *IndexContext) interface{}

	// Visit a parse tree produced by CELParser#Ident.
	VisitIdent(ctx *IdentContext) interface{}

	// Visit a parse tree produced by CELParser#GlobalCall.
	VisitGlobalCall(ctx *GlobalCallContext) interface{}

	// Visit a parse tree produced by CELParser#Nested.
	VisitNested(ctx *NestedContext) interface{}

	// Visit a parse tree produced by CELParser#CreateList.
	VisitCreateList(ctx *CreateListContext) interface{}

	// Visit a parse tree produced by CELParser#CreateStruct.
	VisitCreateStruct(ctx *CreateStructContext) interface{}

	// Visit a parse tree produced by CELParser#CreateMessage.
	VisitCreateMessage(ctx *CreateMessageContext) interface{}

	// Visit a parse tree produced by CELParser#ConstantLiteral.
	VisitConstantLiteral(ctx *ConstantLiteralContext) interface{}

	// Visit a parse tree produced by CELParser#exprList.
	VisitExprList(ctx *ExprListContext) interface{}

	// Visit a parse tree produced by CELParser#listInit.
	VisitListInit(ctx *ListInitContext) interface{}

	// Visit a parse tree produced by CELParser#fieldInitializerList.
	VisitFieldInitializerList(ctx *FieldInitializerListContext) interface{}

	// Visit a parse tree produced by CELParser#optField.
	VisitOptField(ctx *OptFieldContext) interface{}

	// Visit a parse tree produced by CELParser#mapInitializerList.
	VisitMapInitializerList(ctx *MapInitializerListContext) interface{}

	// Visit a parse tree produced by CELParser#SimpleIdentifier.
	VisitSimpleIdentifier(ctx *SimpleIdentifierContext) interface{}

	// Visit a parse tree produced by CELParser#EscapedIdentifier.
	VisitEscapedIdentifier(ctx *EscapedIdentifierContext) interface{}

	// Visit a parse tree produced by CELParser#optExpr.
	VisitOptExpr(ctx *OptExprContext) interface{}

	// Visit a parse tree produced by CELParser#Int.
	VisitInt(ctx *IntContext) interface{}

	// Visit a parse tree produced by CELParser#Uint.
	VisitUint(ctx *UintContext) interface{}

	// Visit a parse tree produced by CELParser#Double.
	VisitDouble(ctx *DoubleContext) interface{}

	// Visit a parse tree produced by CELParser#String.
	VisitString(ctx *StringContext) interface{}

	// Visit a parse tree produced by CELParser#Bytes.
	VisitBytes(ctx *BytesContext) interface{}

	// Visit a parse tree produced by CELParser#BoolTrue.
	VisitBoolTrue(ctx *BoolTrueContext) interface{}

	// Visit a parse tree produced by CELParser#BoolFalse.
	VisitBoolFalse(ctx *BoolFalseContext) interface{}

	// Visit a parse tree produced by CELParser#Null.
	VisitNull(ctx *NullContext) interface{}
}
