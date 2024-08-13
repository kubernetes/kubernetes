// Code generated from /usr/local/google/home/tswadell/go/src/github.com/google/cel-go/parser/gen/CEL.g4 by ANTLR 4.13.1. DO NOT EDIT.

package gen // CEL
import "github.com/antlr4-go/antlr/v4"


type BaseCELVisitor struct {
	*antlr.BaseParseTreeVisitor
}

func (v *BaseCELVisitor) VisitStart(ctx *StartContext) interface{} {
	return v.VisitChildren(ctx)
}

func (v *BaseCELVisitor) VisitExpr(ctx *ExprContext) interface{} {
	return v.VisitChildren(ctx)
}

func (v *BaseCELVisitor) VisitConditionalOr(ctx *ConditionalOrContext) interface{} {
	return v.VisitChildren(ctx)
}

func (v *BaseCELVisitor) VisitConditionalAnd(ctx *ConditionalAndContext) interface{} {
	return v.VisitChildren(ctx)
}

func (v *BaseCELVisitor) VisitRelation(ctx *RelationContext) interface{} {
	return v.VisitChildren(ctx)
}

func (v *BaseCELVisitor) VisitCalc(ctx *CalcContext) interface{} {
	return v.VisitChildren(ctx)
}

func (v *BaseCELVisitor) VisitMemberExpr(ctx *MemberExprContext) interface{} {
	return v.VisitChildren(ctx)
}

func (v *BaseCELVisitor) VisitLogicalNot(ctx *LogicalNotContext) interface{} {
	return v.VisitChildren(ctx)
}

func (v *BaseCELVisitor) VisitNegate(ctx *NegateContext) interface{} {
	return v.VisitChildren(ctx)
}

func (v *BaseCELVisitor) VisitMemberCall(ctx *MemberCallContext) interface{} {
	return v.VisitChildren(ctx)
}

func (v *BaseCELVisitor) VisitSelect(ctx *SelectContext) interface{} {
	return v.VisitChildren(ctx)
}

func (v *BaseCELVisitor) VisitPrimaryExpr(ctx *PrimaryExprContext) interface{} {
	return v.VisitChildren(ctx)
}

func (v *BaseCELVisitor) VisitIndex(ctx *IndexContext) interface{} {
	return v.VisitChildren(ctx)
}

func (v *BaseCELVisitor) VisitIdentOrGlobalCall(ctx *IdentOrGlobalCallContext) interface{} {
	return v.VisitChildren(ctx)
}

func (v *BaseCELVisitor) VisitNested(ctx *NestedContext) interface{} {
	return v.VisitChildren(ctx)
}

func (v *BaseCELVisitor) VisitCreateList(ctx *CreateListContext) interface{} {
	return v.VisitChildren(ctx)
}

func (v *BaseCELVisitor) VisitCreateStruct(ctx *CreateStructContext) interface{} {
	return v.VisitChildren(ctx)
}

func (v *BaseCELVisitor) VisitCreateMessage(ctx *CreateMessageContext) interface{} {
	return v.VisitChildren(ctx)
}

func (v *BaseCELVisitor) VisitConstantLiteral(ctx *ConstantLiteralContext) interface{} {
	return v.VisitChildren(ctx)
}

func (v *BaseCELVisitor) VisitExprList(ctx *ExprListContext) interface{} {
	return v.VisitChildren(ctx)
}

func (v *BaseCELVisitor) VisitListInit(ctx *ListInitContext) interface{} {
	return v.VisitChildren(ctx)
}

func (v *BaseCELVisitor) VisitFieldInitializerList(ctx *FieldInitializerListContext) interface{} {
	return v.VisitChildren(ctx)
}

func (v *BaseCELVisitor) VisitOptField(ctx *OptFieldContext) interface{} {
	return v.VisitChildren(ctx)
}

func (v *BaseCELVisitor) VisitMapInitializerList(ctx *MapInitializerListContext) interface{} {
	return v.VisitChildren(ctx)
}

func (v *BaseCELVisitor) VisitOptExpr(ctx *OptExprContext) interface{} {
	return v.VisitChildren(ctx)
}

func (v *BaseCELVisitor) VisitInt(ctx *IntContext) interface{} {
	return v.VisitChildren(ctx)
}

func (v *BaseCELVisitor) VisitUint(ctx *UintContext) interface{} {
	return v.VisitChildren(ctx)
}

func (v *BaseCELVisitor) VisitDouble(ctx *DoubleContext) interface{} {
	return v.VisitChildren(ctx)
}

func (v *BaseCELVisitor) VisitString(ctx *StringContext) interface{} {
	return v.VisitChildren(ctx)
}

func (v *BaseCELVisitor) VisitBytes(ctx *BytesContext) interface{} {
	return v.VisitChildren(ctx)
}

func (v *BaseCELVisitor) VisitBoolTrue(ctx *BoolTrueContext) interface{} {
	return v.VisitChildren(ctx)
}

func (v *BaseCELVisitor) VisitBoolFalse(ctx *BoolFalseContext) interface{} {
	return v.VisitChildren(ctx)
}

func (v *BaseCELVisitor) VisitNull(ctx *NullContext) interface{} {
	return v.VisitChildren(ctx)
}
