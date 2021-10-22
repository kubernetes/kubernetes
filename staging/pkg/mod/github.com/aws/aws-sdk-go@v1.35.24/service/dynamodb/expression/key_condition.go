package expression

import (
	"fmt"
)

// keyConditionMode specifies the types of the struct KeyConditionBuilder,
// representing the different types of KeyConditions (i.e. And, Or, Between, ...)
type keyConditionMode int

const (
	// unsetKeyCond catches errors for unset KeyConditionBuilder structs
	unsetKeyCond keyConditionMode = iota
	// invalidKeyCond catches errors in the construction of KeyConditionBuilder structs
	invalidKeyCond
	// equalKeyCond represents the Equals KeyCondition
	equalKeyCond
	// lessThanKeyCond represents the Less Than KeyCondition
	lessThanKeyCond
	// lessThanEqualKeyCond represents the Less Than Or Equal To KeyCondition
	lessThanEqualKeyCond
	// greaterThanKeyCond represents the Greater Than KeyCondition
	greaterThanKeyCond
	// greaterThanEqualKeyCond represents the Greater Than Or Equal To KeyCondition
	greaterThanEqualKeyCond
	// andKeyCond represents the Logical And KeyCondition
	andKeyCond
	// betweenKeyCond represents the Between KeyCondition
	betweenKeyCond
	// beginsWithKeyCond represents the Begins With KeyCondition
	beginsWithKeyCond
)

// KeyConditionBuilder represents Key Condition Expressions in DynamoDB.
// KeyConditionBuilders are the building blocks of Expressions.
// More Information at: http://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Query.html#Query.KeyConditionExpressions
type KeyConditionBuilder struct {
	operandList      []OperandBuilder
	keyConditionList []KeyConditionBuilder
	mode             keyConditionMode
}

// KeyEqual returns a KeyConditionBuilder representing the equality clause
// of the two argument OperandBuilders. The resulting KeyConditionBuilder can be
// used as a part of other Key Condition Expressions or as an argument to the
// WithKeyCondition() method for the Builder struct.
//
// Example:
//
//     // keyCondition represents the equal clause of the key "foo" and the
//     // value 5
//     keyCondition := expression.KeyEqual(expression.Key("foo"), expression.Value(5))
//
//     // Used in another Key Condition Expression
//     anotherKeyCondition := expression.Key("partitionKey").Equal(expression.Value("aValue")).And(keyCondition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithKeyCondition(keyCondition)
//
// Expression Equivalent:
//
//     expression.KeyEqual(expression.Key("foo"), expression.Value(5))
//     // Let :five be an ExpressionAttributeValue representing the value 5
//     "foo = :five"
func KeyEqual(keyBuilder KeyBuilder, valueBuilder ValueBuilder) KeyConditionBuilder {
	return KeyConditionBuilder{
		operandList: []OperandBuilder{keyBuilder, valueBuilder},
		mode:        equalKeyCond,
	}
}

// Equal returns a KeyConditionBuilder representing the equality clause of
// the two argument OperandBuilders. The resulting KeyConditionBuilder can be
// used as a part of other Key Condition Expressions or as an argument to the
// WithKeyCondition() method for the Builder struct.
//
// Example:
//
//     // keyCondition represents the equal clause of the key "foo" and the
//     // value 5
//     keyCondition := expression.Key("foo").Equal(expression.Value(5))
//
//     // Used in another Key Condition Expression
//     anotherKeyCondition := expression.Key("partitionKey").Equal(expression.Value("aValue")).And(keyCondition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithKeyCondition(keyCondition)
//
// Expression Equivalent:
//
//     expression.Key("foo").Equal(expression.Value(5))
//     // Let :five be an ExpressionAttributeValue representing the value 5
//     "foo = :five"
func (kb KeyBuilder) Equal(valueBuilder ValueBuilder) KeyConditionBuilder {
	return KeyEqual(kb, valueBuilder)
}

// KeyLessThan returns a KeyConditionBuilder representing the less than
// clause of the two argument OperandBuilders. The resulting KeyConditionBuilder
// can be used as a part of other Key Condition Expressions.
//
// Example:
//
//     // keyCondition represents the less than clause of the key "foo" and the
//     // value 5
//     keyCondition := expression.KeyLessThan(expression.Key("foo"), expression.Value(5))
//
//     // Used in another Key Condition Expression
//     anotherKeyCondition := expression.Key("partitionKey").Equal(expression.Value("aValue")).And(keyCondition)
//
// Expression Equivalent:
//
//     expression.KeyLessThan(expression.Key("foo"), expression.Value(5))
//     // Let :five be an ExpressionAttributeValue representing the value 5
//     "foo < :five"
func KeyLessThan(keyBuilder KeyBuilder, valueBuilder ValueBuilder) KeyConditionBuilder {
	return KeyConditionBuilder{
		operandList: []OperandBuilder{keyBuilder, valueBuilder},
		mode:        lessThanKeyCond,
	}
}

// LessThan returns a KeyConditionBuilder representing the less than clause
// of the two argument OperandBuilders. The resulting KeyConditionBuilder can be
// used as a part of other Key Condition Expressions.
//
// Example:
//
//     // keyCondition represents the less than clause of the key "foo" and the
//     // value 5
//     keyCondition := expression.Key("foo").LessThan(expression.Value(5))
//
//     // Used in another Key Condition Expression
//     anotherKeyCondition := expression.Key("partitionKey").Equal(expression.Value("aValue")).And(keyCondition)
//
// Expression Equivalent:
//
//     expression.Key("foo").LessThan(expression.Value(5))
//     // Let :five be an ExpressionAttributeValue representing the value 5
//     "foo < :five"
func (kb KeyBuilder) LessThan(valueBuilder ValueBuilder) KeyConditionBuilder {
	return KeyLessThan(kb, valueBuilder)
}

// KeyLessThanEqual returns a KeyConditionBuilder representing the less than
// equal to clause of the two argument OperandBuilders. The resulting
// KeyConditionBuilder can be used as a part of other Key Condition Expressions.
//
// Example:
//
//     // keyCondition represents the less than equal to clause of the key
//     // "foo" and the value 5
//     keyCondition := expression.KeyLessThanEqual(expression.Key("foo"), expression.Value(5))
//
//     // Used in another Key Condition Expression
//     anotherKeyCondition := expression.Key("partitionKey").Equal(expression.Value("aValue")).And(keyCondition)
//
// Expression Equivalent:
//
//     expression.KeyLessThanEqual(expression.Key("foo"), expression.Value(5))
//     // Let :five be an ExpressionAttributeValue representing the value 5
//     "foo <= :five"
func KeyLessThanEqual(keyBuilder KeyBuilder, valueBuilder ValueBuilder) KeyConditionBuilder {
	return KeyConditionBuilder{
		operandList: []OperandBuilder{keyBuilder, valueBuilder},
		mode:        lessThanEqualKeyCond,
	}
}

// LessThanEqual returns a KeyConditionBuilder representing the less than
// equal to clause of the two argument OperandBuilders. The resulting
// KeyConditionBuilder can be used as a part of other Key Condition Expressions.
//
// Example:
//
//     // keyCondition represents the less than equal to clause of the key
//     // "foo" and the value 5
//     keyCondition := expression.Key("foo").LessThanEqual(expression.Value(5))
//
//     // Used in another Key Condition Expression
//     anotherKeyCondition := expression.Key("partitionKey").Equal(expression.Value("aValue")).And(keyCondition)
//
// Expression Equivalent:
//
//     expression.Key("foo").LessThanEqual(expression.Value(5))
//     // Let :five be an ExpressionAttributeValue representing the value 5
//     "foo <= :five"
func (kb KeyBuilder) LessThanEqual(valueBuilder ValueBuilder) KeyConditionBuilder {
	return KeyLessThanEqual(kb, valueBuilder)
}

// KeyGreaterThan returns a KeyConditionBuilder representing the greater
// than clause of the two argument OperandBuilders. The resulting
// KeyConditionBuilder can be used as a part of other Key Condition Expressions.
//
// Example:
//
//     // keyCondition represents the greater than clause of the key "foo" and
//     // the value 5
//     keyCondition := expression.KeyGreaterThan(expression.Key("foo"), expression.Value(5))
//
//     // Used in another Key Condition Expression
//     anotherKeyCondition := expression.Key("partitionKey").Equal(expression.Value("aValue")).And(keyCondition)
//
// Expression Equivalent:
//
//     expression.KeyGreaterThan(expression.Key("foo"), expression.Value(5))
//     // Let :five be an ExpressionAttributeValue representing the value 5
//     "foo > :five"
func KeyGreaterThan(keyBuilder KeyBuilder, valueBuilder ValueBuilder) KeyConditionBuilder {
	return KeyConditionBuilder{
		operandList: []OperandBuilder{keyBuilder, valueBuilder},
		mode:        greaterThanKeyCond,
	}
}

// GreaterThan returns a KeyConditionBuilder representing the greater than
// clause of the two argument OperandBuilders. The resulting KeyConditionBuilder
// can be used as a part of other Key Condition Expressions.
//
// Example:
//
//     // key condition represents the greater than clause of the key "foo" and
//     // the value 5
//     keyCondition := expression.Key("foo").GreaterThan(expression.Value(5))
//
//     // Used in another Key Condition Expression
//     anotherKeyCondition := expression.Key("partitionKey").Equal(expression.Value("aValue")).And(keyCondition)
//
// Expression Equivalent:
//
//     expression.Key("foo").GreaterThan(expression.Value(5))
//     // Let :five be an ExpressionAttributeValue representing the value 5
//     "foo > :five"
func (kb KeyBuilder) GreaterThan(valueBuilder ValueBuilder) KeyConditionBuilder {
	return KeyGreaterThan(kb, valueBuilder)
}

// KeyGreaterThanEqual returns a KeyConditionBuilder representing the
// greater than equal to clause of the two argument OperandBuilders. The
// resulting KeyConditionBuilder can be used as a part of other Key Condition
// Expressions.
//
// Example:
//
//     // keyCondition represents the greater than equal to clause of the key
//     // "foo" and the value 5
//     keyCondition := expression.KeyGreaterThanEqual(expression.Key("foo"), expression.Value(5))
//
//     // Used in another Key Condition Expression
//     anotherKeyCondition := expression.Key("partitionKey").Equal(expression.Value("aValue")).And(keyCondition)
//
// Expression Equivalent:
//
//     expression.KeyGreaterThanEqual(expression.Key("foo"), expression.Value(5))
//     // Let :five be an ExpressionAttributeValue representing the value 5
//     "foo >= :five"
func KeyGreaterThanEqual(keyBuilder KeyBuilder, valueBuilder ValueBuilder) KeyConditionBuilder {
	return KeyConditionBuilder{
		operandList: []OperandBuilder{keyBuilder, valueBuilder},
		mode:        greaterThanEqualKeyCond,
	}
}

// GreaterThanEqual returns a KeyConditionBuilder representing the greater
// than equal to clause of the two argument OperandBuilders. The resulting
// KeyConditionBuilder can be used as a part of other Key Condition Expressions.
//
// Example:
//
//     // keyCondition represents the greater than equal to clause of the key
//     // "foo" and the value 5
//     keyCondition := expression.Key("foo").GreaterThanEqual(expression.Value(5))
//
//     // Used in another Key Condition Expression
//     anotherKeyCondition := expression.Key("partitionKey").Equal(expression.Value("aValue")).And(keyCondition)
//
// Expression Equivalent:
//
//     expression.Key("foo").GreaterThanEqual(expression.Value(5))
//     // Let :five be an ExpressionAttributeValue representing the value 5
//     "foo >= :five"
func (kb KeyBuilder) GreaterThanEqual(valueBuilder ValueBuilder) KeyConditionBuilder {
	return KeyGreaterThanEqual(kb, valueBuilder)
}

// KeyAnd returns a KeyConditionBuilder representing the logical AND clause
// of the two argument KeyConditionBuilders. The resulting KeyConditionBuilder
// can be used as an argument to the WithKeyCondition() method for the Builder
// struct.
//
// Example:
//
//     // keyCondition represents the key condition where the partition key
//     // "TeamName" is equal to value "Wildcats" and sort key "Number" is equal
//     // to value 1
//     keyCondition := expression.KeyAnd(expression.Key("TeamName").Equal(expression.Value("Wildcats")), expression.Key("Number").Equal(expression.Value(1)))
//
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithKeyCondition(keyCondition)
//
// Expression Equivalent:
//
//     expression.KeyAnd(expression.Key("TeamName").Equal(expression.Value("Wildcats")), expression.Key("Number").Equal(expression.Value(1)))
//     // Let #NUMBER, :teamName, and :one be ExpressionAttributeName and
//     // ExpressionAttributeValues representing the item attribute "Number",
//     // the value "Wildcats", and the value 1
//     "(TeamName = :teamName) AND (#NUMBER = :one)"
func KeyAnd(left, right KeyConditionBuilder) KeyConditionBuilder {
	if left.mode != equalKeyCond {
		return KeyConditionBuilder{
			mode: invalidKeyCond,
		}
	}
	if right.mode == andKeyCond {
		return KeyConditionBuilder{
			mode: invalidKeyCond,
		}
	}
	return KeyConditionBuilder{
		keyConditionList: []KeyConditionBuilder{left, right},
		mode:             andKeyCond,
	}
}

// And returns a KeyConditionBuilder representing the logical AND clause of
// the two argument KeyConditionBuilders. The resulting KeyConditionBuilder can
// be used as an argument to the WithKeyCondition() method for the Builder
// struct.
//
// Example:
//
//     // keyCondition represents the key condition where the partition key
//     // "TeamName" is equal to value "Wildcats" and sort key "Number" is equal
//     // to value 1
//     keyCondition := expression.Key("TeamName").Equal(expression.Value("Wildcats")).And(expression.Key("Number").Equal(expression.Value(1)))
//
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithKeyCondition(keyCondition)
//
// Expression Equivalent:
//
//     expression.Key("TeamName").Equal(expression.Value("Wildcats")).And(expression.Key("Number").Equal(expression.Value(1)))
//     // Let #NUMBER, :teamName, and :one be ExpressionAttributeName and
//     // ExpressionAttributeValues representing the item attribute "Number",
//     // the value "Wildcats", and the value 1
//     "(TeamName = :teamName) AND (#NUMBER = :one)"
func (kcb KeyConditionBuilder) And(right KeyConditionBuilder) KeyConditionBuilder {
	return KeyAnd(kcb, right)
}

// KeyBetween returns a KeyConditionBuilder representing the result of the
// BETWEEN function in DynamoDB Key Condition Expressions. The resulting
// KeyConditionBuilder can be used as a part of other Key Condition Expressions.
//
// Example:
//
//     // keyCondition represents the boolean key condition of whether the value
//     // of the key "foo" is between values 5 and 10
//     keyCondition := expression.KeyBetween(expression.Key("foo"), expression.Value(5), expression.Value(10))
//
//     // Used in another Key Condition Expression
//     anotherKeyCondition := expression.Key("partitionKey").Equal(expression.Value("aValue")).And(keyCondition)
//
// Expression Equivalent:
//
//     expression.KeyBetween(expression.Key("foo"), expression.Value(5), expression.Value(10))
//     // Let :five and :ten be ExpressionAttributeValues representing the
//     // values 5 and 10 respectively
//     "foo BETWEEN :five AND :ten"
func KeyBetween(keyBuilder KeyBuilder, lower, upper ValueBuilder) KeyConditionBuilder {
	return KeyConditionBuilder{
		operandList: []OperandBuilder{keyBuilder, lower, upper},
		mode:        betweenKeyCond,
	}
}

// Between returns a KeyConditionBuilder representing the result of the
// BETWEEN function in DynamoDB Key Condition Expressions. The resulting
// KeyConditionBuilder can be used as a part of other Key Condition Expressions.
//
// Example:
//
//     // keyCondition represents the boolean key condition of whether the value
//     // of the key "foo" is between values 5 and 10
//     keyCondition := expression.Key("foo").Between(expression.Value(5), expression.Value(10))
//
//     // Used in another Key Condition Expression
//     anotherKeyCondition := expression.Key("partitionKey").Equal(expression.Value("aValue")).And(keyCondition)
//
// Expression Equivalent:
//
//     expression.Key("foo").Between(expression.Value(5), expression.Value(10))
//     // Let :five and :ten be ExpressionAttributeValues representing the
//     // values 5 and 10 respectively
//     "foo BETWEEN :five AND :ten"
func (kb KeyBuilder) Between(lower, upper ValueBuilder) KeyConditionBuilder {
	return KeyBetween(kb, lower, upper)
}

// KeyBeginsWith returns a KeyConditionBuilder representing the result of
// the begins_with function in DynamoDB Key Condition Expressions. The resulting
// KeyConditionBuilder can be used as a part of other Key Condition Expressions.
//
// Example:
//
//     // keyCondition represents the boolean key condition of whether the value
//     // of the key "foo" is begins with the prefix "bar"
//     keyCondition := expression.KeyBeginsWith(expression.Key("foo"), "bar")
//
//     // Used in another Key Condition Expression
//     anotherKeyCondition := expression.Key("partitionKey").Equal(expression.Value("aValue")).And(keyCondition)
//
// Expression Equivalent:
//
//     expression.KeyBeginsWith(expression.Key("foo"), "bar")
//     // Let :bar be an ExpressionAttributeValue representing the value "bar"
//     "begins_with(foo, :bar)"
func KeyBeginsWith(keyBuilder KeyBuilder, prefix string) KeyConditionBuilder {
	valueBuilder := ValueBuilder{
		value: prefix,
	}
	return KeyConditionBuilder{
		operandList: []OperandBuilder{keyBuilder, valueBuilder},
		mode:        beginsWithKeyCond,
	}
}

// BeginsWith returns a KeyConditionBuilder representing the result of the
// begins_with function in DynamoDB Key Condition Expressions. The resulting
// KeyConditionBuilder can be used as a part of other Key Condition Expressions.
//
// Example:
//
//     // keyCondition represents the boolean key condition of whether the value
//     // of the key "foo" is begins with the prefix "bar"
//     keyCondition := expression.Key("foo").BeginsWith("bar")
//
//     // Used in another Key Condition Expression
//     anotherKeyCondition := expression.Key("partitionKey").Equal(expression.Value("aValue")).And(keyCondition)
//
// Expression Equivalent:
//
//     expression.Key("foo").BeginsWith("bar")
//     // Let :bar be an ExpressionAttributeValue representing the value "bar"
//     "begins_with(foo, :bar)"
func (kb KeyBuilder) BeginsWith(prefix string) KeyConditionBuilder {
	return KeyBeginsWith(kb, prefix)
}

// buildTree builds a tree structure of exprNodes based on the tree
// structure of the input KeyConditionBuilder's child KeyConditions/Operands.
// buildTree() satisfies the treeBuilder interface so KeyConditionBuilder can be
// a part of Expression struct.
func (kcb KeyConditionBuilder) buildTree() (exprNode, error) {
	childNodes, err := kcb.buildChildNodes()
	if err != nil {
		return exprNode{}, err
	}
	ret := exprNode{
		children: childNodes,
	}

	switch kcb.mode {
	case equalKeyCond, lessThanKeyCond, lessThanEqualKeyCond, greaterThanKeyCond, greaterThanEqualKeyCond:
		return compareBuildKeyCondition(kcb.mode, ret)
	case andKeyCond:
		return andBuildKeyCondition(kcb, ret)
	case betweenKeyCond:
		return betweenBuildKeyCondition(ret)
	case beginsWithKeyCond:
		return beginsWithBuildKeyCondition(ret)
	case unsetKeyCond:
		return exprNode{}, newUnsetParameterError("buildTree", "KeyConditionBuilder")
	case invalidKeyCond:
		return exprNode{}, fmt.Errorf("buildKeyCondition error: invalid key condition constructed")
	default:
		return exprNode{}, fmt.Errorf("buildKeyCondition error: unsupported mode: %v", kcb.mode)
	}
}

// compareBuildKeyCondition is the function to make exprNodes from Compare
// KeyConditionBuilders. compareBuildKeyCondition is only called by the
// buildKeyCondition method. This function assumes that the argument
// KeyConditionBuilder has the right format.
func compareBuildKeyCondition(keyConditionMode keyConditionMode, node exprNode) (exprNode, error) {
	// Create a string with special characters that can be substituted later: $c
	switch keyConditionMode {
	case equalKeyCond:
		node.fmtExpr = "$c = $c"
	case lessThanKeyCond:
		node.fmtExpr = "$c < $c"
	case lessThanEqualKeyCond:
		node.fmtExpr = "$c <= $c"
	case greaterThanKeyCond:
		node.fmtExpr = "$c > $c"
	case greaterThanEqualKeyCond:
		node.fmtExpr = "$c >= $c"
	default:
		return exprNode{}, fmt.Errorf("build compare key condition error: unsupported mode: %v", keyConditionMode)
	}

	return node, nil
}

// andBuildKeyCondition is the function to make exprNodes from And
// KeyConditionBuilders. andBuildKeyCondition is only called by the
// buildKeyCondition method. This function assumes that the argument
// KeyConditionBuilder has the right format.
func andBuildKeyCondition(keyConditionBuilder KeyConditionBuilder, node exprNode) (exprNode, error) {
	if len(keyConditionBuilder.keyConditionList) == 0 && len(keyConditionBuilder.operandList) == 0 {
		return exprNode{}, newInvalidParameterError("andBuildKeyCondition", "KeyConditionBuilder")
	}
	// create a string with escaped characters to substitute them with proper
	// aliases during runtime
	node.fmtExpr = "($c) AND ($c)"

	return node, nil
}

// betweenBuildKeyCondition is the function to make exprNodes from Between
// KeyConditionBuilders. betweenBuildKeyCondition is only called by the
// buildKeyCondition method. This function assumes that the argument
// KeyConditionBuilder has the right format.
func betweenBuildKeyCondition(node exprNode) (exprNode, error) {
	// Create a string with special characters that can be substituted later: $c
	node.fmtExpr = "$c BETWEEN $c AND $c"

	return node, nil
}

// beginsWithBuildKeyCondition is the function to make exprNodes from
// BeginsWith KeyConditionBuilders. beginsWithBuildKeyCondition is only
// called by the buildKeyCondition method. This function assumes that the argument
// KeyConditionBuilder has the right format.
func beginsWithBuildKeyCondition(node exprNode) (exprNode, error) {
	// Create a string with special characters that can be substituted later: $c
	node.fmtExpr = "begins_with ($c, $c)"

	return node, nil
}

// buildChildNodes creates the list of the child exprNodes. This avoids
// duplication of code amongst the various buildConditions.
func (kcb KeyConditionBuilder) buildChildNodes() ([]exprNode, error) {
	childNodes := make([]exprNode, 0, len(kcb.keyConditionList)+len(kcb.operandList))
	for _, keyCondition := range kcb.keyConditionList {
		node, err := keyCondition.buildTree()
		if err != nil {
			return []exprNode{}, err
		}
		childNodes = append(childNodes, node)
	}
	for _, operand := range kcb.operandList {
		ope, err := operand.BuildOperand()
		if err != nil {
			return []exprNode{}, err
		}
		childNodes = append(childNodes, ope.exprNode)
	}

	return childNodes, nil
}
