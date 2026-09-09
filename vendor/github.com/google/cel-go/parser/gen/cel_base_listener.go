// Code generated from /usr/local/google/home/jdtatum/github/cel-go/parser/gen/CEL.g4 by ANTLR 4.13.1. DO NOT EDIT.

package gen // CEL
import "github.com/antlr4-go/antlr/v4"

// BaseCELListener is a complete listener for a parse tree produced by CELParser.
type BaseCELListener struct{}

var _ CELListener = &BaseCELListener{}

// VisitTerminal is called when a terminal node is visited.
func (s *BaseCELListener) VisitTerminal(node antlr.TerminalNode) {}

// VisitErrorNode is called when an error node is visited.
func (s *BaseCELListener) VisitErrorNode(node antlr.ErrorNode) {}

// EnterEveryRule is called when any rule is entered.
func (s *BaseCELListener) EnterEveryRule(ctx antlr.ParserRuleContext) {}

// ExitEveryRule is called when any rule is exited.
func (s *BaseCELListener) ExitEveryRule(ctx antlr.ParserRuleContext) {}

// EnterStart is called when production start is entered.
func (s *BaseCELListener) EnterStart(ctx *StartContext) {}

// ExitStart is called when production start is exited.
func (s *BaseCELListener) ExitStart(ctx *StartContext) {}

// EnterExpr is called when production expr is entered.
func (s *BaseCELListener) EnterExpr(ctx *ExprContext) {}

// ExitExpr is called when production expr is exited.
func (s *BaseCELListener) ExitExpr(ctx *ExprContext) {}

// EnterConditionalOr is called when production conditionalOr is entered.
func (s *BaseCELListener) EnterConditionalOr(ctx *ConditionalOrContext) {}

// ExitConditionalOr is called when production conditionalOr is exited.
func (s *BaseCELListener) ExitConditionalOr(ctx *ConditionalOrContext) {}

// EnterConditionalAnd is called when production conditionalAnd is entered.
func (s *BaseCELListener) EnterConditionalAnd(ctx *ConditionalAndContext) {}

// ExitConditionalAnd is called when production conditionalAnd is exited.
func (s *BaseCELListener) ExitConditionalAnd(ctx *ConditionalAndContext) {}

// EnterRelation is called when production relation is entered.
func (s *BaseCELListener) EnterRelation(ctx *RelationContext) {}

// ExitRelation is called when production relation is exited.
func (s *BaseCELListener) ExitRelation(ctx *RelationContext) {}

// EnterCalc is called when production calc is entered.
func (s *BaseCELListener) EnterCalc(ctx *CalcContext) {}

// ExitCalc is called when production calc is exited.
func (s *BaseCELListener) ExitCalc(ctx *CalcContext) {}

// EnterMemberExpr is called when production MemberExpr is entered.
func (s *BaseCELListener) EnterMemberExpr(ctx *MemberExprContext) {}

// ExitMemberExpr is called when production MemberExpr is exited.
func (s *BaseCELListener) ExitMemberExpr(ctx *MemberExprContext) {}

// EnterLogicalNot is called when production LogicalNot is entered.
func (s *BaseCELListener) EnterLogicalNot(ctx *LogicalNotContext) {}

// ExitLogicalNot is called when production LogicalNot is exited.
func (s *BaseCELListener) ExitLogicalNot(ctx *LogicalNotContext) {}

// EnterNegate is called when production Negate is entered.
func (s *BaseCELListener) EnterNegate(ctx *NegateContext) {}

// ExitNegate is called when production Negate is exited.
func (s *BaseCELListener) ExitNegate(ctx *NegateContext) {}

// EnterMemberCall is called when production MemberCall is entered.
func (s *BaseCELListener) EnterMemberCall(ctx *MemberCallContext) {}

// ExitMemberCall is called when production MemberCall is exited.
func (s *BaseCELListener) ExitMemberCall(ctx *MemberCallContext) {}

// EnterSelect is called when production Select is entered.
func (s *BaseCELListener) EnterSelect(ctx *SelectContext) {}

// ExitSelect is called when production Select is exited.
func (s *BaseCELListener) ExitSelect(ctx *SelectContext) {}

// EnterPrimaryExpr is called when production PrimaryExpr is entered.
func (s *BaseCELListener) EnterPrimaryExpr(ctx *PrimaryExprContext) {}

// ExitPrimaryExpr is called when production PrimaryExpr is exited.
func (s *BaseCELListener) ExitPrimaryExpr(ctx *PrimaryExprContext) {}

// EnterIndex is called when production Index is entered.
func (s *BaseCELListener) EnterIndex(ctx *IndexContext) {}

// ExitIndex is called when production Index is exited.
func (s *BaseCELListener) ExitIndex(ctx *IndexContext) {}

// EnterIdent is called when production Ident is entered.
func (s *BaseCELListener) EnterIdent(ctx *IdentContext) {}

// ExitIdent is called when production Ident is exited.
func (s *BaseCELListener) ExitIdent(ctx *IdentContext) {}

// EnterGlobalCall is called when production GlobalCall is entered.
func (s *BaseCELListener) EnterGlobalCall(ctx *GlobalCallContext) {}

// ExitGlobalCall is called when production GlobalCall is exited.
func (s *BaseCELListener) ExitGlobalCall(ctx *GlobalCallContext) {}

// EnterNested is called when production Nested is entered.
func (s *BaseCELListener) EnterNested(ctx *NestedContext) {}

// ExitNested is called when production Nested is exited.
func (s *BaseCELListener) ExitNested(ctx *NestedContext) {}

// EnterCreateList is called when production CreateList is entered.
func (s *BaseCELListener) EnterCreateList(ctx *CreateListContext) {}

// ExitCreateList is called when production CreateList is exited.
func (s *BaseCELListener) ExitCreateList(ctx *CreateListContext) {}

// EnterCreateStruct is called when production CreateStruct is entered.
func (s *BaseCELListener) EnterCreateStruct(ctx *CreateStructContext) {}

// ExitCreateStruct is called when production CreateStruct is exited.
func (s *BaseCELListener) ExitCreateStruct(ctx *CreateStructContext) {}

// EnterCreateMessage is called when production CreateMessage is entered.
func (s *BaseCELListener) EnterCreateMessage(ctx *CreateMessageContext) {}

// ExitCreateMessage is called when production CreateMessage is exited.
func (s *BaseCELListener) ExitCreateMessage(ctx *CreateMessageContext) {}

// EnterConstantLiteral is called when production ConstantLiteral is entered.
func (s *BaseCELListener) EnterConstantLiteral(ctx *ConstantLiteralContext) {}

// ExitConstantLiteral is called when production ConstantLiteral is exited.
func (s *BaseCELListener) ExitConstantLiteral(ctx *ConstantLiteralContext) {}

// EnterExprList is called when production exprList is entered.
func (s *BaseCELListener) EnterExprList(ctx *ExprListContext) {}

// ExitExprList is called when production exprList is exited.
func (s *BaseCELListener) ExitExprList(ctx *ExprListContext) {}

// EnterListInit is called when production listInit is entered.
func (s *BaseCELListener) EnterListInit(ctx *ListInitContext) {}

// ExitListInit is called when production listInit is exited.
func (s *BaseCELListener) ExitListInit(ctx *ListInitContext) {}

// EnterFieldInitializerList is called when production fieldInitializerList is entered.
func (s *BaseCELListener) EnterFieldInitializerList(ctx *FieldInitializerListContext) {}

// ExitFieldInitializerList is called when production fieldInitializerList is exited.
func (s *BaseCELListener) ExitFieldInitializerList(ctx *FieldInitializerListContext) {}

// EnterOptField is called when production optField is entered.
func (s *BaseCELListener) EnterOptField(ctx *OptFieldContext) {}

// ExitOptField is called when production optField is exited.
func (s *BaseCELListener) ExitOptField(ctx *OptFieldContext) {}

// EnterMapInitializerList is called when production mapInitializerList is entered.
func (s *BaseCELListener) EnterMapInitializerList(ctx *MapInitializerListContext) {}

// ExitMapInitializerList is called when production mapInitializerList is exited.
func (s *BaseCELListener) ExitMapInitializerList(ctx *MapInitializerListContext) {}

// EnterSimpleIdentifier is called when production SimpleIdentifier is entered.
func (s *BaseCELListener) EnterSimpleIdentifier(ctx *SimpleIdentifierContext) {}

// ExitSimpleIdentifier is called when production SimpleIdentifier is exited.
func (s *BaseCELListener) ExitSimpleIdentifier(ctx *SimpleIdentifierContext) {}

// EnterEscapedIdentifier is called when production EscapedIdentifier is entered.
func (s *BaseCELListener) EnterEscapedIdentifier(ctx *EscapedIdentifierContext) {}

// ExitEscapedIdentifier is called when production EscapedIdentifier is exited.
func (s *BaseCELListener) ExitEscapedIdentifier(ctx *EscapedIdentifierContext) {}

// EnterOptExpr is called when production optExpr is entered.
func (s *BaseCELListener) EnterOptExpr(ctx *OptExprContext) {}

// ExitOptExpr is called when production optExpr is exited.
func (s *BaseCELListener) ExitOptExpr(ctx *OptExprContext) {}

// EnterInt is called when production Int is entered.
func (s *BaseCELListener) EnterInt(ctx *IntContext) {}

// ExitInt is called when production Int is exited.
func (s *BaseCELListener) ExitInt(ctx *IntContext) {}

// EnterUint is called when production Uint is entered.
func (s *BaseCELListener) EnterUint(ctx *UintContext) {}

// ExitUint is called when production Uint is exited.
func (s *BaseCELListener) ExitUint(ctx *UintContext) {}

// EnterDouble is called when production Double is entered.
func (s *BaseCELListener) EnterDouble(ctx *DoubleContext) {}

// ExitDouble is called when production Double is exited.
func (s *BaseCELListener) ExitDouble(ctx *DoubleContext) {}

// EnterString is called when production String is entered.
func (s *BaseCELListener) EnterString(ctx *StringContext) {}

// ExitString is called when production String is exited.
func (s *BaseCELListener) ExitString(ctx *StringContext) {}

// EnterBytes is called when production Bytes is entered.
func (s *BaseCELListener) EnterBytes(ctx *BytesContext) {}

// ExitBytes is called when production Bytes is exited.
func (s *BaseCELListener) ExitBytes(ctx *BytesContext) {}

// EnterBoolTrue is called when production BoolTrue is entered.
func (s *BaseCELListener) EnterBoolTrue(ctx *BoolTrueContext) {}

// ExitBoolTrue is called when production BoolTrue is exited.
func (s *BaseCELListener) ExitBoolTrue(ctx *BoolTrueContext) {}

// EnterBoolFalse is called when production BoolFalse is entered.
func (s *BaseCELListener) EnterBoolFalse(ctx *BoolFalseContext) {}

// ExitBoolFalse is called when production BoolFalse is exited.
func (s *BaseCELListener) ExitBoolFalse(ctx *BoolFalseContext) {}

// EnterNull is called when production Null is entered.
func (s *BaseCELListener) EnterNull(ctx *NullContext) {}

// ExitNull is called when production Null is exited.
func (s *BaseCELListener) ExitNull(ctx *NullContext) {}
