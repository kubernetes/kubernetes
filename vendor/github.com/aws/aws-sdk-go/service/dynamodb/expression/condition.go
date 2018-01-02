package expression

import (
	"fmt"
	"strings"
)

// conditionMode specifies the types of the struct conditionBuilder,
// representing the different types of Conditions (i.e. And, Or, Between, ...)
type conditionMode int

const (
	// unsetCond catches errors for unset ConditionBuilder structs
	unsetCond conditionMode = iota
	// equalCond represents the Equals Condition
	equalCond
	// notEqualCond represents the Not Equals Condition
	notEqualCond
	// lessThanCond represents the LessThan Condition
	lessThanCond
	// lessThanEqualCond represents the LessThanOrEqual Condition
	lessThanEqualCond
	// greaterThanCond represents the GreaterThan Condition
	greaterThanCond
	// greaterThanEqualCond represents the GreaterThanEqual Condition
	greaterThanEqualCond
	// andCond represents the Logical And Condition
	andCond
	// orCond represents the Logical Or Condition
	orCond
	// notCond represents the Logical Not Condition
	notCond
	// betweenCond represents the Between Condition
	betweenCond
	// inCond represents the In Condition
	inCond
	// attrExistsCond represents the Attribute Exists Condition
	attrExistsCond
	// attrNotExistsCond represents the Attribute Not Exists Condition
	attrNotExistsCond
	// attrTypeCond represents the Attribute Type Condition
	attrTypeCond
	// beginsWithCond represents the Begins With Condition
	beginsWithCond
	// containsCond represents the Contains Condition
	containsCond
)

// DynamoDBAttributeType specifies the type of an DynamoDB item attribute. This
// enum is used in the AttributeType() function in order to be explicit about
// the DynamoDB type that is being checked and ensure compile time checks.
// More Informatin at http://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Expressions.OperatorsAndFunctions.html#Expressions.OperatorsAndFunctions.Functions
type DynamoDBAttributeType string

const (
	// String represents the DynamoDB String type
	String DynamoDBAttributeType = "S"
	// StringSet represents the DynamoDB String Set type
	StringSet = "SS"
	// Number represents the DynamoDB Number type
	Number = "N"
	// NumberSet represents the DynamoDB Number Set type
	NumberSet = "NS"
	// Binary represents the DynamoDB Binary type
	Binary = "B"
	// BinarySet represents the DynamoDB Binary Set type
	BinarySet = "BS"
	// Boolean represents the DynamoDB Boolean type
	Boolean = "BOOL"
	// Null represents the DynamoDB Null type
	Null = "NULL"
	// List represents the DynamoDB List type
	List = "L"
	// Map represents the DynamoDB Map type
	Map = "M"
)

// ConditionBuilder represents Condition Expressions and Filter Expressions
// in DynamoDB. ConditionBuilders are one of the building blocks of the Builder
// struct. Since Filter Expressions support all the same functions and formats
// as Condition Expressions, ConditionBuilders represents both types of
// Expressions.
// More Information at: http://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Expressions.ConditionExpressions.html
// More Information on Filter Expressions: http://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Query.html#Query.FilterExpression
type ConditionBuilder struct {
	operandList   []OperandBuilder
	conditionList []ConditionBuilder
	mode          conditionMode
}

// Equal returns a ConditionBuilder representing the equality clause of the two
// argument OperandBuilders. The resulting ConditionBuilder can be used as a
// part of other Condition Expressions or as an argument to the WithCondition()
// method for the Builder struct.
//
// Example:
//
//     // condition represents the equal clause of the item attribute "foo" and
//     // the value 5
//     condition := expression.Equal(expression.Name("foo"), expression.Value(5))
//
//     // Used in another Condition Expression
//     anotherCondition := expression.Not(condition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithCondition(condition)
//
// Expression Equivalent:
//
//     expression.Equal(expression.Name("foo"), expression.Value(5))
//     // Let :five be an ExpressionAttributeValue representing the value 5
//     "foo = :five"
func Equal(left, right OperandBuilder) ConditionBuilder {
	return ConditionBuilder{
		operandList: []OperandBuilder{left, right},
		mode:        equalCond,
	}
}

// Equal returns a ConditionBuilder representing the equality clause of the two
// argument OperandBuilders. The resulting ConditionBuilder can be used as a
// part of other Condition Expressions or as an argument to the WithCondition()
// method for the Builder struct.
//
// Example:
//
//     // condition represents the equal clause of the item attribute "foo" and
//     // the value 5
//     condition := expression.Name("foo").Equal(expression.Value(5))
//
//     // Used in another Condition Expression
//     anotherCondition := expression.Not(condition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithCondition(condition)
//
// Expression Equivalent:
//
//     expression.Name("foo").Equal(expression.Value(5))
//     // Let :five be an ExpressionAttributeValue representing the value 5
//     "foo = :five"
func (nb NameBuilder) Equal(right OperandBuilder) ConditionBuilder {
	return Equal(nb, right)
}

// Equal returns a ConditionBuilder representing the equality clause of the two
// argument OperandBuilders. The resulting ConditionBuilder can be used as a
// part of other Condition Expressions or as an argument to the WithCondition()
// method for the Builder struct.
//
// Example:
//
//     // condition represents the equal clause of the item attribute "foo" and
//     // the value 5
//     condition := expression.Value(5).Equal(expression.Name("foo"))
//
//     // Used in another Condition Expression
//     anotherCondition := expression.Not(condition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithCondition(condition)
//
// Expression Equivalent:
//
//     expression.Value(5).Equal(expression.Name("foo"))
//     // Let :five be an ExpressionAttributeValue representing the value 5
//     ":five = foo"
func (vb ValueBuilder) Equal(right OperandBuilder) ConditionBuilder {
	return Equal(vb, right)
}

// Equal returns a ConditionBuilder representing the equality clause of the two
// argument OperandBuilders. The resulting ConditionBuilder can be used as a
// part of other Condition Expressions or as an argument to the WithCondition()
// method for the Builder struct.
//
// Example:
//
//     // condition represents the equal clause of the size of the item
//     // attribute "foo" and the value 5
//     condition := expression.Size(expression.Name("foo")).Equal(expression.Value(5))
//
//     // Used in another Condition Expression
//     anotherCondition := expression.Not(condition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithCondition(condition)
//
// Expression Equivalent:
//
//     expression.Size(expression.Name("foo")).Equal(expression.Value(5))
//     // Let :five be an ExpressionAttributeValue representing the value 5
//     "size (foo) = :five"
func (sb SizeBuilder) Equal(right OperandBuilder) ConditionBuilder {
	return Equal(sb, right)
}

// NotEqual returns a ConditionBuilder representing the not equal clause of the
// two argument OperandBuilders. The resulting ConditionBuilder can be used as a
// part of other Condition Expressions or as an argument to the WithCondition()
// method for the Builder struct.
//
// Example:
//
//     // condition represents the not equal clause of the item attribute "foo"
//     // and the value 5
//     condition := expression.NotEqual(expression.Name("foo"), expression.Value(5))
//
//     // Used in another Condition Expression
//     anotherCondition := expression.Not(condition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithCondition(condition)
//
// Expression Equivalent:
//
//     expression.NotEqual(expression.Name("foo"), expression.Value(5))
//     // Let :five be an ExpressionAttributeValue representing the value 5
//     "foo <> :five"
func NotEqual(left, right OperandBuilder) ConditionBuilder {
	return ConditionBuilder{
		operandList: []OperandBuilder{left, right},
		mode:        notEqualCond,
	}
}

// NotEqual returns a ConditionBuilder representing the not equal clause of the
// two argument OperandBuilders. The resulting ConditionBuilder can be used as a
// part of other Condition Expressions or as an argument to the WithCondition()
// method for the Builder struct.
//
// Example:
//
//     // condition represents the not equal clause of the item attribute "foo"
//     // and the value 5
//     condition := expression.Name("foo").NotEqual(expression.Value(5))
//
//     // Used in another Condition Expression
//     anotherCondition := expression.Not(condition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithCondition(condition)
//
// Expression Equivalent:
//
//     expression.Name("foo").NotEqual(expression.Value(5))
//     // Let :five be an ExpressionAttributeValue representing the value 5
//     "foo <> :five"
func (nb NameBuilder) NotEqual(right OperandBuilder) ConditionBuilder {
	return NotEqual(nb, right)
}

// NotEqual returns a ConditionBuilder representing the not equal clause of the
// two argument OperandBuilders. The resulting ConditionBuilder can be used as a
// part of other Condition Expressions or as an argument to the WithCondition()
// method for the Builder struct.
//
// Example:
//
//     // condition represents the not equal clause of the item attribute "foo"
//     // and the value 5
//     condition := expression.Value(5).NotEqual(expression.Name("foo"))
//
//     // Used in another Condition Expression
//     anotherCondition := expression.Not(condition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithCondition(condition)
//
// Expression Equivalent:
//
//     expression.Value(5).NotEqual(expression.Name("foo"))
//     // Let :five be an ExpressionAttributeValue representing the value 5
//     ":five <> foo"
func (vb ValueBuilder) NotEqual(right OperandBuilder) ConditionBuilder {
	return NotEqual(vb, right)
}

// NotEqual returns a ConditionBuilder representing the not equal clause of the
// two argument OperandBuilders. The resulting ConditionBuilder can be used as a
// part of other Condition Expressions or as an argument to the WithCondition()
// method for the Builder struct.
//
// Example:
//
//     // condition represents the not equal clause of the size of the item
//     // attribute "foo" and the value 5
//     condition := expression.Size(expression.Name("foo")).NotEqual(expression.Value(5))
//
//     // Used in another Condition Expression
//     anotherCondition := expression.Not(condition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithCondition(condition)
//
// Expression Equivalent:
//
//     expression.Size(expression.Name("foo")).NotEqual(expression.Value(5))
//     // Let :five be an ExpressionAttributeValue representing the value 5
//     "size (foo) <> :five"
func (sb SizeBuilder) NotEqual(right OperandBuilder) ConditionBuilder {
	return NotEqual(sb, right)
}

// LessThan returns a ConditionBuilder representing the less than clause of the
// two argument OperandBuilders. The resulting ConditionBuilder can be used as a
// part of other Condition Expressions or as an argument to the WithCondition()
// method for the Builder struct.
//
// Example:
//
//     // condition represents the less than clause of the item attribute "foo"
//     // and the value 5
//     condition := expression.LessThan(expression.Name("foo"), expression.Value(5))
//
//     // Used in another Condition Expression
//     anotherCondition := expression.Not(condition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithCondition(condition)
//
// Expression Equivalent:
//
//     expression.LessThan(expression.Name("foo"), expression.Value(5))
//     // Let :five be an ExpressionAttributeValue representing the value 5
//     "foo < :five"
func LessThan(left, right OperandBuilder) ConditionBuilder {
	return ConditionBuilder{
		operandList: []OperandBuilder{left, right},
		mode:        lessThanCond,
	}
}

// LessThan returns a ConditionBuilder representing the less than clause of the
// two argument OperandBuilders. The resulting ConditionBuilder can be used as a
// part of other Condition Expressions or as an argument to the WithCondition()
// method for the Builder struct.
//
// Example:
//
//     // condition represents the less than clause of the item attribute "foo"
//     // and the value 5
//     condition := expression.Name("foo").LessThan(expression.Value(5))
//
//     // Used in another Condition Expression
//     anotherCondition := expression.Not(condition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithCondition(condition)
//
// Expression Equivalent:
//
//     expression.Name("foo").LessThan(expression.Value(5))
//     // Let :five be an ExpressionAttributeValue representing the value 5
//     "foo < :five"
func (nb NameBuilder) LessThan(right OperandBuilder) ConditionBuilder {
	return LessThan(nb, right)
}

// LessThan returns a ConditionBuilder representing the less than clause of the
// two argument OperandBuilders. The resulting ConditionBuilder can be used as a
// part of other Condition Expressions or as an argument to the WithCondition()
// method for the Builder struct.
//
// Example:
//
//     // condition represents the less than clause of the item attribute "foo"
//     // and the value 5
//     condition := expression.Value(5).LessThan(expression.Name("foo"))
//
//     // Used in another Condition Expression
//     anotherCondition := expression.Not(condition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithCondition(condition)
//
// Expression Equivalent:
//
//     expression.Value(5).LessThan(expression.Name("foo"))
//     // Let :five be an ExpressionAttributeValue representing the value 5
//     ":five < foo"
func (vb ValueBuilder) LessThan(right OperandBuilder) ConditionBuilder {
	return LessThan(vb, right)
}

// LessThan returns a ConditionBuilder representing the less than clause of the
// two argument OperandBuilders. The resulting ConditionBuilder can be used as a
// part of other Condition Expressions or as an argument to the WithCondition()
// method for the Builder struct.
//
// Example:
//
//     // condition represents the less than clause of the size of the item
//     // attribute "foo" and the value 5
//     condition := expression.Size(expression.Name("foo")).LessThan(expression.Value(5))
//
//     // Used in another Condition Expression
//     anotherCondition := expression.Not(condition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithCondition(condition)
//
// Expression Equivalent:
//
//     expression.Size(expression.Name("foo")).LessThan(expression.Value(5))
//     // Let :five be an ExpressionAttributeValue representing the value 5
//     "size (foo) < :five"
func (sb SizeBuilder) LessThan(right OperandBuilder) ConditionBuilder {
	return LessThan(sb, right)
}

// LessThanEqual returns a ConditionBuilder representing the less than equal to
// clause of the two argument OperandBuilders. The resulting ConditionBuilder
// can be used as a part of other Condition Expressions or as an argument to the
// WithCondition() method for the Builder struct.
//
// Example:
//
//     // condition represents the less than equal to clause of the item
//     // attribute "foo" and the value 5
//     condition := expression.LessThanEqual(expression.Name("foo"), expression.Value(5))
//
//     // Used in another Condition Expression
//     anotherCondition := expression.Not(condition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithCondition(condition)
//
// Expression Equivalent:
//
//     expression.LessThanEqual(expression.Name("foo"), expression.Value(5))
//     // Let :five be an ExpressionAttributeValue representing the value 5
//     "foo <= :five"
func LessThanEqual(left, right OperandBuilder) ConditionBuilder {
	return ConditionBuilder{
		operandList: []OperandBuilder{left, right},
		mode:        lessThanEqualCond,
	}
}

// LessThanEqual returns a ConditionBuilder representing the less than equal to
// clause of the two argument OperandBuilders. The resulting ConditionBuilder
// can be used as a part of other Condition Expressions or as an argument to the
// WithCondition() method for the Builder struct.
//
// Example:
//
//     // condition represents the less than equal to clause of the item
//     // attribute "foo" and the value 5
//     condition := expression.Name("foo").LessThanEqual(expression.Value(5))
//
//     // Used in another Condition Expression
//     anotherCondition := expression.Not(condition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithCondition(condition)
//
// Expression Equivalent:
//
//     expression.Name("foo").LessThanEqual(expression.Value(5))
//     // Let :five be an ExpressionAttributeValue representing the value 5
//     "foo <= :five"
func (nb NameBuilder) LessThanEqual(right OperandBuilder) ConditionBuilder {
	return LessThanEqual(nb, right)
}

// LessThanEqual returns a ConditionBuilder representing the less than equal to
// clause of the two argument OperandBuilders. The resulting ConditionBuilder
// can be used as a part of other Condition Expressions or as an argument to the
// WithCondition() method for the Builder struct.
//
// Example:
//
//     // condition represents the less than equal to clause of the item
//     // attribute "foo" and the value 5
//     condition := expression.Value(5).LessThanEqual(expression.Name("foo"))
//
//     // Used in another Condition Expression
//     anotherCondition := expression.Not(condition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithCondition(condition)
//
// Expression Equivalent:
//
//     expression.Value(5).LessThanEqual(expression.Name("foo"))
//     // Let :five be an ExpressionAttributeValue representing the value 5
//     ":five <= foo"
func (vb ValueBuilder) LessThanEqual(right OperandBuilder) ConditionBuilder {
	return LessThanEqual(vb, right)
}

// LessThanEqual returns a ConditionBuilder representing the less than equal to
// clause of the two argument OperandBuilders. The resulting ConditionBuilder
// can be used as a part of other Condition Expressions or as an argument to the
// WithCondition() method for the Builder struct.
//
// Example:
//
//     // condition represents the less than equal to clause of the size of the
//     // item attribute "foo" and the value 5
//     condition := expression.Size(expression.Name("foo")).LessThanEqual(expression.Value(5))
//
//     // Used in another Condition Expression
//     anotherCondition := expression.Not(condition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithCondition(condition)
//
// Expression Equivalent:
//
//     expression.Size(expression.Name("foo")).LessThanEqual(expression.Value(5))
//     // Let :five be an ExpressionAttributeValue representing the value 5
//     "size (foo) <= :five"
func (sb SizeBuilder) LessThanEqual(right OperandBuilder) ConditionBuilder {
	return LessThanEqual(sb, right)
}

// GreaterThan returns a ConditionBuilder representing the greater than clause
// of the two argument OperandBuilders. The resulting ConditionBuilder can be
// used as a part of other Condition Expressions or as an argument to the
// WithCondition() method for the Builder struct.
//
// Example:
//
//     // condition represents the greater than clause of the item attribute
//     // "foo" and the value 5
//     condition := expression.GreaterThan(expression.Name("foo"), expression.Value(5))
//
//     // Used in another Condition Expression
//     anotherCondition := expression.Not(condition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithCondition(condition)
//
// Expression Equivalent:
//
//     expression.GreaterThan(expression.Name("foo"), expression.Value(5))
//     // Let :five be an ExpressionAttributeValue representing the value 5
//     "foo > :five"
func GreaterThan(left, right OperandBuilder) ConditionBuilder {
	return ConditionBuilder{
		operandList: []OperandBuilder{left, right},
		mode:        greaterThanCond,
	}
}

// GreaterThan returns a ConditionBuilder representing the greater than clause
// of the two argument OperandBuilders. The resulting ConditionBuilder can be
// used as a part of other Condition Expressions or as an argument to the
// WithCondition() method for the Builder struct.
//
// Example:
//
//     // condition represents the greater than clause of the item attribute
//     // "foo" and the value 5
//     condition := expression.Name("foo").GreaterThan(expression.Value(5))
//
//     // Used in another Condition Expression
//     anotherCondition := expression.Not(condition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithCondition(condition)
//
// Expression Equivalent:
//
//     expression.Name("foo").GreaterThan(expression.Value(5))
//     // Let :five be an ExpressionAttributeValue representing the value 5
//     "foo > :five"
func (nb NameBuilder) GreaterThan(right OperandBuilder) ConditionBuilder {
	return GreaterThan(nb, right)
}

// GreaterThan returns a ConditionBuilder representing the greater than clause
// of the two argument OperandBuilders. The resulting ConditionBuilder can be
// used as a part of other Condition Expressions or as an argument to the
// WithCondition() method for the Builder struct.
//
// Example:
//
//     // condition represents the greater than clause of the item attribute
//     // "foo" and the value 5
//     condition := expression.Value(5).GreaterThan(expression.Name("foo"))
//
//     // Used in another Condition Expression
//     anotherCondition := expression.Not(condition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithCondition(condition)
//
// Expression Equivalent:
//
//     expression.Value(5).GreaterThan(expression.Name("foo"))
//     // Let :five be an ExpressionAttributeValue representing the value 5
//     ":five > foo"
func (vb ValueBuilder) GreaterThan(right OperandBuilder) ConditionBuilder {
	return GreaterThan(vb, right)
}

// GreaterThan returns a ConditionBuilder representing the greater than
// clause of the two argument OperandBuilders. The resulting ConditionBuilder
// can be used as a part of other Condition Expressions or as an argument to the
// WithCondition() method for the Builder struct.
//
// Example:
//
//     // condition represents the greater than clause of the size of the item
//     // attribute "foo" and the value 5
//     condition := expression.Size(expression.Name("foo")).GreaterThan(expression.Value(5))
//
//     // Used in another Condition Expression
//     anotherCondition := expression.Not(condition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithCondition(condition)
//
// Expression Equivalent:
//
//     expression.Size(expression.Name("foo")).GreaterThan(expression.Value(5))
//     // Let :five be an ExpressionAttributeValue representing the value 5
//     "size (foo) > :five"
func (sb SizeBuilder) GreaterThan(right OperandBuilder) ConditionBuilder {
	return GreaterThan(sb, right)
}

// GreaterThanEqual returns a ConditionBuilder representing the greater than
// equal to clause of the two argument OperandBuilders. The resulting
// ConditionBuilder can be used as a part of other Condition Expressions or as
// an argument to the WithCondition() method for the Builder struct.
//
// Example:
//
//     // condition represents the greater than equal to clause of the item
//     // attribute "foo" and the value 5
//     condition := expression.GreaterThanEqual(expression.Name("foo"), expression.Value(5))
//
//     // Used in another Condition Expression
//     anotherCondition := expression.Not(condition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithCondition(condition)
//
// Expression Equivalent:
//
//     expression.GreaterThanEqual(expression.Name("foo"), expression.Value(5))
//     // Let :five be an ExpressionAttributeValue representing the value 5
//     "foo >= :five"
func GreaterThanEqual(left, right OperandBuilder) ConditionBuilder {
	return ConditionBuilder{
		operandList: []OperandBuilder{left, right},
		mode:        greaterThanEqualCond,
	}
}

// GreaterThanEqual returns a ConditionBuilder representing the greater than
// equal to clause of the two argument OperandBuilders. The resulting
// ConditionBuilder can be used as a part of other Condition Expressions or as
// an argument to the WithCondition() method for the Builder struct.
//
// Example:
//
//     // condition represents the greater than equal to clause of the item
//     // attribute "foo" and the value 5
//     condition := expression.Name("foo").GreaterThanEqual(expression.Value(5))
//
//     // Used in another Condition Expression
//     anotherCondition := expression.Not(condition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithCondition(condition)
//
// Expression Equivalent:
//
//     expression.Name("foo").GreaterThanEqual(expression.Value(5))
//     // Let :five be an ExpressionAttributeValue representing the value 5
//     "foo >= :five"
func (nb NameBuilder) GreaterThanEqual(right OperandBuilder) ConditionBuilder {
	return GreaterThanEqual(nb, right)
}

// GreaterThanEqual returns a ConditionBuilder representing the greater than
// equal to clause of the two argument OperandBuilders. The resulting
// ConditionBuilder can be used as a part of other Condition Expressions or as
// an argument to the WithCondition() method for the Builder struct.
//
// Example:
//
//     // condition represents the greater than equal to clause of the item
//     // attribute "foo" and the value 5
//     condition := expression.Value(5).GreaterThanEqual(expression.Name("foo"))
//
//     // Used in another Condition Expression
//     anotherCondition := expression.Not(condition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithCondition(condition)
//
// Expression Equivalent:
//
//     expression.Value(5).GreaterThanEqual(expression.Name("foo"))
//     // Let :five be an ExpressionAttributeValue representing the value 5
//     ":five >= foo"
func (vb ValueBuilder) GreaterThanEqual(right OperandBuilder) ConditionBuilder {
	return GreaterThanEqual(vb, right)
}

// GreaterThanEqual returns a ConditionBuilder representing the greater than
// equal to clause of the two argument OperandBuilders. The resulting
// ConditionBuilder can be used as a part of other Condition Expressions or as
// an argument to the WithCondition() method for the Builder struct.
//
// Example:
//
//     // condition represents the greater than equal to clause of the size of
//     // the item attribute "foo" and the value 5
//     condition := expression.Size(expression.Name("foo")).GreaterThanEqual(expression.Value(5))
//
//     // Used in another Condition Expression
//     anotherCondition := expression.Not(condition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithCondition(condition)
//
// Expression Equivalent:
//
//     expression.Size(expression.Name("foo")).GreaterThanEqual(expression.Value(5))
//     // Let :five be an ExpressionAttributeValue representing the value 5
//     "size (foo) >= :five"
func (sb SizeBuilder) GreaterThanEqual(right OperandBuilder) ConditionBuilder {
	return GreaterThanEqual(sb, right)
}

// And returns a ConditionBuilder representing the logical AND clause of the
// argument ConditionBuilders. The resulting ConditionBuilder can be used as a
// part of other Condition Expressions or as an argument to the WithCondition()
// method for the Builder struct. Note that And() can take a variadic number of
// ConditionBuilders as arguments.
//
// Example:
//
//     // condition represents the condition where the item attribute "Name" is
//     // equal to value "Generic Name" AND the item attribute "Age" is less
//     // than value 40
//     condition := expression.And(expression.Name("Name").Equal(expression.Value("Generic Name")), expression.Name("Age").LessThan(expression.Value(40)))
//
//     // Used in another Condition Expression
//     anotherCondition := expression.Not(condition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithCondition(condition)
//
// Expression Equivalent:
//
//     expression.And(expression.Name("Name").Equal(expression.Value("Generic Name")), expression.Name("Age").LessThan(expression.Value(40)))
//     // Let #NAME, :name, and :forty be ExpressionAttributeName and
//     // ExpressionAttributeValues representing the item attribute "Name", the
//     // value "Generic Name", and the value 40
//     "(#NAME = :name) AND (Age < :forty)"
func And(left, right ConditionBuilder, other ...ConditionBuilder) ConditionBuilder {
	other = append([]ConditionBuilder{left, right}, other...)
	return ConditionBuilder{
		conditionList: other,
		mode:          andCond,
	}
}

// And returns a ConditionBuilder representing the logical AND clause of the
// argument ConditionBuilders. The resulting ConditionBuilder can be used as a
// part of other Condition Expressions or as an argument to the WithCondition()
// method for the Builder struct. Note that And() can take a variadic number of
// ConditionBuilders as arguments.
//
// Example:
//
//     // condition represents the condition where the item attribute "Name" is
//     // equal to value "Generic Name" AND the item attribute "Age" is less
//     // than value 40
//     condition := expression.Name("Name").Equal(expression.Value("Generic Name")).And(expression.Name("Age").LessThan(expression.Value(40)))
//
//     // Used in another Condition Expression
//     anotherCondition := expression.Not(condition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithCondition(condition)
//
// Expression Equivalent:
//
//     expression.Name("Name").Equal(expression.Value("Generic Name")).And(expression.Name("Age").LessThan(expression.Value(40)))
//     // Let #NAME, :name, and :forty be ExpressionAttributeName and
//     // ExpressionAttributeValues representing the item attribute "Name", the
//     // value "Generic Name", and the value 40
//     "(#NAME = :name) AND (Age < :forty)"
func (cb ConditionBuilder) And(right ConditionBuilder, other ...ConditionBuilder) ConditionBuilder {
	return And(cb, right, other...)
}

// Or returns a ConditionBuilder representing the logical OR clause of the
// argument ConditionBuilders. The resulting ConditionBuilder can be used as a
// part of other Condition Expressions or as an argument to the WithCondition()
// method for the Builder struct. Note that Or() can take a variadic number of
// ConditionBuilders as arguments.
//
// Example:
//
//     // condition represents the condition where the item attribute "Price" is
//     // less than the value 100 OR the item attribute "Rating" is greater than
//     // the value 8
//     condition := expression.Or(expression.Name("Price").Equal(expression.Value(100)), expression.Name("Rating").LessThan(expression.Value(8)))
//
//     // Used in another Condition Expression
//     anotherCondition := expression.Not(condition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithCondition(condition)
//
// Expression Equivalent:
//
//     expression.Or(expression.Name("Price").Equal(expression.Value(100)), expression.Name("Rating").LessThan(expression.Value(8)))
//     // Let :price and :rating be ExpressionAttributeValues representing the
//     // the value 100 and value 8 respectively
//     "(Price < :price) OR (Rating > :rating)"
func Or(left, right ConditionBuilder, other ...ConditionBuilder) ConditionBuilder {
	other = append([]ConditionBuilder{left, right}, other...)
	return ConditionBuilder{
		conditionList: other,
		mode:          orCond,
	}
}

// Or returns a ConditionBuilder representing the logical OR clause of the
// argument ConditionBuilders. The resulting ConditionBuilder can be used as a
// part of other Condition Expressions or as an argument to the WithCondition()
// method for the Builder struct. Note that Or() can take a variadic number of
// ConditionBuilders as arguments.
//
// Example:
//
//     // condition represents the condition where the item attribute "Price" is
//     // less than the value 100 OR the item attribute "Rating" is greater than
//     // the value 8
//     condition := expression.Name("Price").Equal(expression.Value(100)).Or(expression.Name("Rating").LessThan(expression.Value(8)))
//
//     // Used in another Condition Expression
//     anotherCondition := expression.Not(condition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithCondition(condition)
//
// Expression Equivalent:
//
//     expression.Name("Price").Equal(expression.Value(100)).Or(expression.Name("Rating").LessThan(expression.Value(8)))
//     // Let :price and :rating be ExpressionAttributeValues representing the
//     // the value 100 and value 8 respectively
//     "(Price < :price) OR (Rating > :rating)"
func (cb ConditionBuilder) Or(right ConditionBuilder, other ...ConditionBuilder) ConditionBuilder {
	return Or(cb, right, other...)
}

// Not returns a ConditionBuilder representing the logical NOT clause of the
// argument ConditionBuilder. The resulting ConditionBuilder can be used as a
// part of other Condition Expressions or as an argument to the WithCondition()
// method for the Builder struct.
//
// Example:
//
//     // condition represents the condition where the item attribute "Name"
//     // does not begin with "test"
//     condition := expression.Not(expression.Name("Name").BeginsWith("test"))
//
//     // Used in another Condition Expression
//     anotherCondition := expression.Not(condition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithCondition(condition)
//
// Expression Equivalent:
//
//     expression.Not(expression.Name("Name").BeginsWith("test"))
//     // Let :prefix be an ExpressionAttributeValue representing the value
//     // "test"
//     "NOT (begins_with (:prefix))"
func Not(conditionBuilder ConditionBuilder) ConditionBuilder {
	return ConditionBuilder{
		conditionList: []ConditionBuilder{conditionBuilder},
		mode:          notCond,
	}
}

// Not returns a ConditionBuilder representing the logical NOT clause of the
// argument ConditionBuilder. The resulting ConditionBuilder can be used as a
// part of other Condition Expressions or as an argument to the WithCondition()
// method for the Builder struct.
//
// Example:
//
//     // condition represents the condition where the item attribute "Name"
//     // does not begin with "test"
//     condition := expression.Name("Name").BeginsWith("test").Not()
//
//     // Used in another Condition Expression
//     anotherCondition := expression.Not(condition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithCondition(condition)
//
// Expression Equivalent:
//
//     expression.Name("Name").BeginsWith("test").Not()
//     // Let :prefix be an ExpressionAttributeValue representing the value
//     // "test"
//     "NOT (begins_with (:prefix))"
func (cb ConditionBuilder) Not() ConditionBuilder {
	return Not(cb)
}

// Between returns a ConditionBuilder representing the result of the
// BETWEEN function in DynamoDB Condition Expressions. The resulting
// ConditionBuilder can be used as a part of other Condition Expressions or as
// an argument to the WithCondition() method for the Builder struct.
//
// Example:
//
//     // condition represents the condition where the value of the item
//     // attribute "Rating" is between values 5 and 10
//     condition := expression.Between(expression.Name("Rating"), expression.Value(5), expression.Value(10))
//
//     // Used in another Condition Expression
//     anotherCondition := expression.Not(condition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithCondition(condition)
//
// Expression Equivalent:
//
//     expression.Between(expression.Name("Rating"), expression.Value(5), expression.Value(10))
//     // Let :five and :ten be ExpressionAttributeValues representing the value
//     // 5 and the value 10
//     "Rating BETWEEN :five AND :ten"
func Between(op, lower, upper OperandBuilder) ConditionBuilder {
	return ConditionBuilder{
		operandList: []OperandBuilder{op, lower, upper},
		mode:        betweenCond,
	}
}

// Between returns a ConditionBuilder representing the result of the
// BETWEEN function in DynamoDB Condition Expressions. The resulting
// ConditionBuilder can be used as a part of other Condition Expressions or as
// an argument to the WithCondition() method for the Builder struct.
//
// Example:
//
//     // condition represents the condition where the value of the item
//     // attribute "Rating" is between values 5 and 10
//     condition := expression.Name("Rating").Between(expression.Value(5), expression.Value(10))
//
//     // Used in another Condition Expression
//     anotherCondition := expression.Not(condition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithCondition(condition)
//
// Expression Equivalent:
//
//     expression.Name("Rating").Between(expression.Value(5), expression.Value(10))
//     // Let :five and :ten be ExpressionAttributeValues representing the value
//     // 5 and the value 10
//     "Rating BETWEEN :five AND :ten"
func (nb NameBuilder) Between(lower, upper OperandBuilder) ConditionBuilder {
	return Between(nb, lower, upper)
}

// Between returns a ConditionBuilder representing the result of the
// BETWEEN function in DynamoDB Condition Expressions. The resulting
// ConditionBuilder can be used as a part of other Condition Expressions or as
// an argument to the WithCondition() method for the Builder struct.
//
// Example:
//
//     // condition represents the condition where the value 6 is between values
//     // 5 and 10
//     condition := expression.Value(6).Between(expression.Value(5), expression.Value(10))
//
//     // Used in another Condition Expression
//     anotherCondition := expression.Not(condition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithCondition(condition)
//
// Expression Equivalent:
//
//     expression.Value(6).Between(expression.Value(5), expression.Value(10))
//     // Let :six, :five and :ten be ExpressionAttributeValues representing the
//     // values 6, 5, and 10 respectively
//     ":six BETWEEN :five AND :ten"
func (vb ValueBuilder) Between(lower, upper OperandBuilder) ConditionBuilder {
	return Between(vb, lower, upper)
}

// Between returns a ConditionBuilder representing the result of the
// BETWEEN function in DynamoDB Condition Expressions. The resulting
// ConditionBuilder can be used as a part of other Condition Expressions or as
// an argument to the WithCondition() method for the Builder struct.
//
// Example:
//
//     // condition represents the condition where the size of the item
//     // attribute "InviteList" is between values 5 and 10
//     condition := expression.Size(expression.Name("InviteList")).Between(expression.Value(5), expression.Value(10))
//
//     // Used in another Condition Expression
//     anotherCondition := expression.Not(condition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithCondition(condition)
//
// Expression Equivalent:
//
//     expression.Size(expression.Name("InviteList")).Between(expression.Value(5), expression.Value(10))
//     // Let :five and :ten be ExpressionAttributeValues representing the value
//     // 5 and the value 10
//     "size (InviteList) BETWEEN :five AND :ten"
func (sb SizeBuilder) Between(lower, upper OperandBuilder) ConditionBuilder {
	return Between(sb, lower, upper)
}

// In returns a ConditionBuilder representing the result of the IN function
// in DynamoDB Condition Expressions. The resulting ConditionBuilder can be used
// as a part of other Condition Expressions or as an argument to the
// WithCondition() method for the Builder struct.
//
// Example:
//
//     // condition represents the condition where the value of the item
//     // attribute "Color" is checked against the list of colors "red",
//     // "green", and "blue".
//     condition := expression.In(expression.Name("Color"), expression.Value("red"), expression.Value("green"), expression.Value("blue"))
//
//     // Used in another Condition Expression
//     anotherCondition := expression.Not(condition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithCondition(condition)
//
// Expression Equivalent:
//
//     expression.In(expression.Name("Color"), expression.Value("red"), expression.Value("green"), expression.Value("blue"))
//     // Let :red, :green, :blue be ExpressionAttributeValues representing the
//     // values "red", "green", and "blue" respectively
//     "Color IN (:red, :green, :blue)"
func In(left, right OperandBuilder, other ...OperandBuilder) ConditionBuilder {
	other = append([]OperandBuilder{left, right}, other...)
	return ConditionBuilder{
		operandList: other,
		mode:        inCond,
	}
}

// In returns a ConditionBuilder representing the result of the IN function
// in DynamoDB Condition Expressions. The resulting ConditionBuilder can be used
// as a part of other Condition Expressions or as an argument to the
// WithCondition() method for the Builder struct.
//
// Example:
//
//     // condition represents the condition where the value of the item
//     // attribute "Color" is checked against the list of colors "red",
//     // "green", and "blue".
//     condition := expression.Name("Color").In(expression.Value("red"), expression.Value("green"), expression.Value("blue"))
//
//     // Used in another Condition Expression
//     anotherCondition := expression.Not(condition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithCondition(condition)
//
// Expression Equivalent:
//
//     expression.Name("Color").In(expression.Value("red"), expression.Value("green"), expression.Value("blue"))
//     // Let :red, :green, :blue be ExpressionAttributeValues representing the
//     // values "red", "green", and "blue" respectively
//     "Color IN (:red, :green, :blue)"
func (nb NameBuilder) In(right OperandBuilder, other ...OperandBuilder) ConditionBuilder {
	return In(nb, right, other...)
}

// In returns a ConditionBuilder representing the result of the IN function
// TODO change this one
// in DynamoDB Condition Expressions. The resulting ConditionBuilder can be used
// as a part of other Condition Expressions or as an argument to the
// WithCondition() method for the Builder struct.
//
// Example:
//
//     // condition represents the condition where the value "yellow" is checked
//     // against the list of colors "red", "green", and "blue".
//     condition := expression.Value("yellow").In(expression.Value("red"), expression.Value("green"), expression.Value("blue"))
//
//     // Used in another Condition Expression
//     anotherCondition := expression.Not(condition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithCondition(condition)
//
// Expression Equivalent:
//
//     expression.Value("yellow").In(expression.Value("red"), expression.Value("green"), expression.Value("blue"))
//     // Let :yellow, :red, :green, :blue be ExpressionAttributeValues
//     // representing the values "yellow", "red", "green", and "blue"
//     // respectively
//     ":yellow IN (:red, :green, :blue)"
func (vb ValueBuilder) In(right OperandBuilder, other ...OperandBuilder) ConditionBuilder {
	return In(vb, right, other...)
}

// In returns a ConditionBuilder representing the result of the IN function
// in DynamoDB Condition Expressions. The resulting ConditionBuilder can be used
// as a part of other Condition Expressions or as an argument to the
// WithCondition() method for the Builder struct.
//
// Example:
//
//     // condition represents the condition where the size of the item
//     // attribute "Donuts" is checked against the list of numbers 12, 24, and
//     // 36.
//     condition := expression.Size(expression.Name("Donuts")).In(expression.Value(12), expression.Value(24), expression.Value(36))
//
//     // Used in another Condition Expression
//     anotherCondition := expression.Not(condition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithCondition(condition)
//
// Expression Equivalent:
//
//     expression.Size(expression.Name("Donuts")).In(expression.Value(12), expression.Value(24), expression.Value(36))
//     // Let :dozen, :twoDozen, :threeDozen be ExpressionAttributeValues
//     // representing the values 12, 24, and 36 respectively
//     "size (Donuts) IN (12, 24, 36)"
func (sb SizeBuilder) In(right OperandBuilder, other ...OperandBuilder) ConditionBuilder {
	return In(sb, right, other...)
}

// AttributeExists returns a ConditionBuilder representing the result of the
// attribute_exists function in DynamoDB Condition Expressions. The resulting
// ConditionBuilder can be used as a part of other Condition Expressions or as
// an argument to the WithCondition() method for the Builder struct.
//
// Example:
//
//     // condition represents the boolean condition of whether the item
//     // attribute "Age" exists or not
//     condition := expression.AttributeExists(expression.Name("Age"))
//
//     // Used in another Condition Expression
//     anotherCondition := expression.Not(condition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithCondition(condition)
//
// Expression Equivalent:
//
//     expression.AttributeExists(expression.Name("Age"))
//     "attribute_exists (Age))"
func AttributeExists(nameBuilder NameBuilder) ConditionBuilder {
	return ConditionBuilder{
		operandList: []OperandBuilder{nameBuilder},
		mode:        attrExistsCond,
	}
}

// AttributeExists returns a ConditionBuilder representing the result of the
// attribute_exists function in DynamoDB Condition Expressions. The resulting
// ConditionBuilder can be used as a part of other Condition Expressions or as
// an argument to the WithCondition() method for the Builder struct.
//
// Example:
//
//     // condition represents the boolean condition of whether the item
//     // attribute "Age" exists or not
//     condition := expression.Name("Age").AttributeExists()
//
//     // Used in another Condition Expression
//     anotherCondition := expression.Not(condition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithCondition(condition)
//
// Expression Equivalent:
//
//     expression.Name("Age").AttributeExists()
//     "attribute_exists (Age))"
func (nb NameBuilder) AttributeExists() ConditionBuilder {
	return AttributeExists(nb)
}

// AttributeNotExists returns a ConditionBuilder representing the result of
// the attribute_not_exists function in DynamoDB Condition Expressions. The
// resulting ConditionBuilder can be used as a part of other Condition
// Expressions or as an argument to the WithCondition() method for the Builder
// struct.
//
// Example:
//
//     // condition represents the boolean condition of whether the item
//     // attribute "Age" exists or not
//     condition := expression.AttributeNotExists(expression.Name("Age"))
//
//     // Used in another Condition Expression
//     anotherCondition := expression.Not(condition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithCondition(condition)
//
// Expression Equivalent:
//
//     expression.AttributeNotExists(expression.Name("Age"))
//     "attribute_not_exists (Age))"
func AttributeNotExists(nameBuilder NameBuilder) ConditionBuilder {
	return ConditionBuilder{
		operandList: []OperandBuilder{nameBuilder},
		mode:        attrNotExistsCond,
	}
}

// AttributeNotExists returns a ConditionBuilder representing the result of
// the attribute_not_exists function in DynamoDB Condition Expressions. The
// resulting ConditionBuilder can be used as a part of other Condition
// Expressions or as an argument to the WithCondition() method for the Builder
// struct.
//
// Example:
//
//     // condition represents the boolean condition of whether the item
//     // attribute "Age" exists or not
//     condition := expression.Name("Age").AttributeNotExists()
//
//     // Used in another Condition Expression
//     anotherCondition := expression.Not(condition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithCondition(condition)
//
// Expression Equivalent:
//
//     expression.Name("Age").AttributeNotExists()
//     "attribute_not_exists (Age))"
func (nb NameBuilder) AttributeNotExists() ConditionBuilder {
	return AttributeNotExists(nb)
}

// AttributeType returns a ConditionBuilder representing the result of the
// attribute_type function in DynamoDB Condition Expressions. The DynamoDB types
// are represented by the type DynamoDBAttributeType. The resulting
// ConditionBuilder can be used as a part of other Condition Expressions or as
// an argument to the WithCondition() method for the Builder struct.
//
// Example:
//
//     // condition represents the boolean condition of whether the item
//     // attribute "Age" has the DynamoDB type Number or not
//     condition := expression.AttributeType(expression.Name("Age"), expression.Number)
//
//     // Used in another Condition Expression
//     anotherCondition := expression.Not(condition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithCondition(condition)
//
// Expression Equivalent:
//
//     expression.AttributeType(expression.Name("Age"), expression.Number)
//     // Let :type be an ExpressionAttributeValue representing the value "N"
//     "attribute_type (Age, :type)"
func AttributeType(nameBuilder NameBuilder, attributeType DynamoDBAttributeType) ConditionBuilder {
	v := ValueBuilder{
		value: string(attributeType),
	}
	return ConditionBuilder{
		operandList: []OperandBuilder{nameBuilder, v},
		mode:        attrTypeCond,
	}
}

// AttributeType returns a ConditionBuilder representing the result of the
// attribute_type function in DynamoDB Condition Expressions. The DynamoDB types
// are represented by the type DynamoDBAttributeType. The resulting
// ConditionBuilder can be used as a part of other Condition Expressions or as
// an argument to the WithCondition() method for the Builder struct.
//
// Example:
//
//     // condition represents the boolean condition of whether the item
//     // attribute "Age" has the DynamoDB type Number or not
//     condition := expression.Name("Age").AttributeType(expression.Number)
//
//     // Used in another Condition Expression
//     anotherCondition := expression.Not(condition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithCondition(condition)
//
// Expression Equivalent:
//
//     expression.Name("Age").AttributeType(expression.Number)
//     // Let :type be an ExpressionAttributeValue representing the value "N"
//     "attribute_type (Age, :type)"
func (nb NameBuilder) AttributeType(attributeType DynamoDBAttributeType) ConditionBuilder {
	return AttributeType(nb, attributeType)
}

// BeginsWith returns a ConditionBuilder representing the result of the
// begins_with function in DynamoDB Condition Expressions. The resulting
// ConditionBuilder can be used as a part of other Condition Expressions or as
// an argument to the WithCondition() method for the Builder struct.
//
// Example:
//
//     // condition represents the boolean condition of whether the item
//     // attribute "CodeName" starts with the substring "Ben"
//     condition := expression.BeginsWith(expression.Name("CodeName"), "Ben")
//
//     // Used in another Condition Expression
//     anotherCondition := expression.Not(condition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithCondition(condition)
//
// Expression Equivalent:
//
//     expression.BeginsWith(expression.Name("CodeName"), "Ben")
//     // Let :ben be an ExpressionAttributeValue representing the value "Ben"
//     "begins_with (CodeName, :ben)"
func BeginsWith(nameBuilder NameBuilder, prefix string) ConditionBuilder {
	v := ValueBuilder{
		value: prefix,
	}
	return ConditionBuilder{
		operandList: []OperandBuilder{nameBuilder, v},
		mode:        beginsWithCond,
	}
}

// BeginsWith returns a ConditionBuilder representing the result of the
// begins_with function in DynamoDB Condition Expressions. The resulting
// ConditionBuilder can be used as a part of other Condition Expressions or as
// an argument to the WithCondition() method for the Builder struct.
//
// Example:
//
//     // condition represents the boolean condition of whether the item
//     // attribute "CodeName" starts with the substring "Ben"
//     condition := expression.Name("CodeName").BeginsWith("Ben")
//
//     // Used in another Condition Expression
//     anotherCondition := expression.Not(condition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithCondition(condition)
//
// Expression Equivalent:
//
//     expression.Name("CodeName").BeginsWith("Ben")
//     // Let :ben be an ExpressionAttributeValue representing the value "Ben"
//     "begins_with (CodeName, :ben)"
func (nb NameBuilder) BeginsWith(prefix string) ConditionBuilder {
	return BeginsWith(nb, prefix)
}

// Contains returns a ConditionBuilder representing the result of the
// contains function in DynamoDB Condition Expressions. The resulting
// ConditionBuilder can be used as a part of other Condition Expressions or as
// an argument to the WithCondition() method for the Builder struct.
//
// Example:
//
//     // condition represents the boolean condition of whether the item
//     // attribute "InviteList" has the value "Ben"
//     condition := expression.Contains(expression.Name("InviteList"), expression.Value("Ben"))
//
//     // Used in another Condition Expression
//     anotherCondition := expression.Not(condition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithCondition(condition)
//
// Expression Equivalent:
//
//     expression.Contains(expression.Name("InviteList"), expression.Value("Ben"))
//     // Let :ben be an ExpressionAttributeValue representing the value "Ben"
//     "contains (InviteList, :ben)"
func Contains(nameBuilder NameBuilder, substr string) ConditionBuilder {
	v := ValueBuilder{
		value: substr,
	}
	return ConditionBuilder{
		operandList: []OperandBuilder{nameBuilder, v},
		mode:        containsCond,
	}
}

// Contains returns a ConditionBuilder representing the result of the
// contains function in DynamoDB Condition Expressions. The resulting
// ConditionBuilder can be used as a part of other Condition Expressions or as
// an argument to the WithCondition() method for the Builder struct.
//
// Example:
//
//     // condition represents the boolean condition of whether the item
//     // attribute "InviteList" has the value "Ben"
//     condition := expression.Name("InviteList").Contains(expression.Value("Ben"))
//
//     // Used in another Condition Expression
//     anotherCondition := expression.Not(condition)
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithCondition(condition)
//
// Expression Equivalent:
//
//     expression.Name("InviteList").Contains(expression.Value("Ben"))
//     // Let :ben be an ExpressionAttributeValue representing the value "Ben"
//     "contains (InviteList, :ben)"
func (nb NameBuilder) Contains(substr string) ConditionBuilder {
	return Contains(nb, substr)
}

// buildTree builds a tree structure of exprNodes based on the tree
// structure of the input ConditionBuilder's child ConditionBuilders and
// OperandBuilders. buildTree() satisfies the treeBuilder interface so
// ConditionBuilder can be a part of Builder and Expression struct.
func (cb ConditionBuilder) buildTree() (exprNode, error) {
	childNodes, err := cb.buildChildNodes()
	if err != nil {
		return exprNode{}, err
	}
	ret := exprNode{
		children: childNodes,
	}

	switch cb.mode {
	case equalCond, notEqualCond, lessThanCond, lessThanEqualCond, greaterThanCond, greaterThanEqualCond:
		return compareBuildCondition(cb.mode, ret)
	case andCond, orCond:
		return compoundBuildCondition(cb, ret)
	case notCond:
		return notBuildCondition(ret)
	case betweenCond:
		return betweenBuildCondition(ret)
	case inCond:
		return inBuildCondition(cb, ret)
	case attrExistsCond:
		return attrExistsBuildCondition(ret)
	case attrNotExistsCond:
		return attrNotExistsBuildCondition(ret)
	case attrTypeCond:
		return attrTypeBuildCondition(ret)
	case beginsWithCond:
		return beginsWithBuildCondition(ret)
	case containsCond:
		return containsBuildCondition(ret)
	case unsetCond:
		return exprNode{}, newUnsetParameterError("buildTree", "ConditionBuilder")
	default:
		return exprNode{}, fmt.Errorf("build condition error: unsupported mode: %v", cb.mode)
	}
}

// compareBuildCondition is the function to make exprNodes from Compare
// ConditionBuilders. compareBuildCondition is only called by the
// buildTree method. This function assumes that the argument ConditionBuilder
// has the right format.
func compareBuildCondition(conditionMode conditionMode, node exprNode) (exprNode, error) {
	// Create a string with special characters that can be substituted later: $c
	switch conditionMode {
	case equalCond:
		node.fmtExpr = "$c = $c"
	case notEqualCond:
		node.fmtExpr = "$c <> $c"
	case lessThanCond:
		node.fmtExpr = "$c < $c"
	case lessThanEqualCond:
		node.fmtExpr = "$c <= $c"
	case greaterThanCond:
		node.fmtExpr = "$c > $c"
	case greaterThanEqualCond:
		node.fmtExpr = "$c >= $c"
	default:
		return exprNode{}, fmt.Errorf("build compare condition error: unsupported mode: %v", conditionMode)
	}

	return node, nil
}

// compoundBuildCondition is the function to make exprNodes from And/Or
// ConditionBuilders. compoundBuildCondition is only called by the
// buildTree method. This function assumes that the argument ConditionBuilder
// has the right format.
func compoundBuildCondition(conditionBuilder ConditionBuilder, node exprNode) (exprNode, error) {
	// create a string with escaped characters to substitute them with proper
	// aliases during runtime
	var mode string
	switch conditionBuilder.mode {
	case andCond:
		mode = " AND "
	case orCond:
		mode = " OR "
	default:
		return exprNode{}, fmt.Errorf("build compound condition error: unsupported mode: %v", conditionBuilder.mode)
	}
	node.fmtExpr = "($c)" + strings.Repeat(mode+"($c)", len(conditionBuilder.conditionList)-1)

	return node, nil
}

// notBuildCondition is the function to make exprNodes from Not
// ConditionBuilders. notBuildCondition is only called by the
// buildTree method. This function assumes that the argument ConditionBuilder
// has the right format.
func notBuildCondition(node exprNode) (exprNode, error) {
	// create a string with escaped characters to substitute them with proper
	// aliases during runtime
	node.fmtExpr = "NOT ($c)"

	return node, nil
}

// betweenBuildCondition is the function to make exprNodes from Between
// ConditionBuilders. BuildCondition is only called by the
// buildTree method. This function assumes that the argument ConditionBuilder
// has the right format.
func betweenBuildCondition(node exprNode) (exprNode, error) {
	// Create a string with special characters that can be substituted later: $c
	node.fmtExpr = "$c BETWEEN $c AND $c"

	return node, nil
}

// inBuildCondition is the function to make exprNodes from In
// ConditionBuilders. inBuildCondition is only called by the
// buildTree method. This function assumes that the argument ConditionBuilder
// has the right format.
func inBuildCondition(conditionBuilder ConditionBuilder, node exprNode) (exprNode, error) {
	// Create a string with special characters that can be substituted later: $c
	node.fmtExpr = "$c IN ($c" + strings.Repeat(", $c", len(conditionBuilder.operandList)-2) + ")"

	return node, nil
}

// attrExistsBuildCondition is the function to make exprNodes from
// AttrExistsCond ConditionBuilders. attrExistsBuildCondition is only
// called by the buildTree method. This function assumes that the argument
// ConditionBuilder has the right format.
func attrExistsBuildCondition(node exprNode) (exprNode, error) {
	// Create a string with special characters that can be substituted later: $c
	node.fmtExpr = "attribute_exists ($c)"

	return node, nil
}

// attrNotExistsBuildCondition is the function to make exprNodes from
// AttrNotExistsCond ConditionBuilders. attrNotExistsBuildCondition is only
// called by the buildTree method. This function assumes that the argument
// ConditionBuilder has the right format.
func attrNotExistsBuildCondition(node exprNode) (exprNode, error) {
	// Create a string with special characters that can be substituted later: $c
	node.fmtExpr = "attribute_not_exists ($c)"

	return node, nil
}

// attrTypeBuildCondition is the function to make exprNodes from AttrTypeCond
// ConditionBuilders. attrTypeBuildCondition is only called by the
// buildTree method. This function assumes that the argument
// ConditionBuilder has the right format.
func attrTypeBuildCondition(node exprNode) (exprNode, error) {
	// Create a string with special characters that can be substituted later: $c
	node.fmtExpr = "attribute_type ($c, $c)"

	return node, nil
}

// beginsWithBuildCondition is the function to make exprNodes from
// BeginsWithCond ConditionBuilders. beginsWithBuildCondition is only
// called by the buildTree method. This function assumes that the argument
// ConditionBuilder has the right format.
func beginsWithBuildCondition(node exprNode) (exprNode, error) {
	// Create a string with special characters that can be substituted later: $c
	node.fmtExpr = "begins_with ($c, $c)"

	return node, nil
}

// containsBuildCondition is the function to make exprNodes from
// ContainsCond ConditionBuilders. containsBuildCondition is only
// called by the buildTree method. This function assumes that the argument
// ConditionBuilder has the right format.
func containsBuildCondition(node exprNode) (exprNode, error) {
	// Create a string with special characters that can be substituted later: $c
	node.fmtExpr = "contains ($c, $c)"

	return node, nil
}

// buildChildNodes creates the list of the child exprNodes. This avoids
// duplication of code amongst the various buildTree functions.
func (cb ConditionBuilder) buildChildNodes() ([]exprNode, error) {
	childNodes := make([]exprNode, 0, len(cb.conditionList)+len(cb.operandList))
	for _, condition := range cb.conditionList {
		node, err := condition.buildTree()
		if err != nil {
			return []exprNode{}, err
		}
		childNodes = append(childNodes, node)
	}
	for _, ope := range cb.operandList {
		operand, err := ope.BuildOperand()
		if err != nil {
			return []exprNode{}, err
		}
		childNodes = append(childNodes, operand.exprNode)
	}

	return childNodes, nil
}
