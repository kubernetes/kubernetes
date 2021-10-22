package expression

import (
	"fmt"
	"strings"

	"github.com/aws/aws-sdk-go/service/dynamodb"
	"github.com/aws/aws-sdk-go/service/dynamodb/dynamodbattribute"
)

// ValueBuilder represents an item attribute value operand and implements the
// OperandBuilder interface. Methods and functions in the package take
// ValueBuilder as an argument and establishes relationships between operands.
// ValueBuilder should only be initialized using the function Value().
//
// Example:
//
//     // Create a ValueBuilder representing the string "aValue"
//     valueBuilder := expression.Value("aValue")
type ValueBuilder struct {
	value interface{}
}

// NameBuilder represents a name of a top level item attribute or a nested
// attribute. Since NameBuilder represents a DynamoDB Operand, it implements the
// OperandBuilder interface. Methods and functions in the package take
// NameBuilder as an argument and establishes relationships between operands.
// NameBuilder should only be initialized using the function Name().
//
// Example:
//
//     // Create a NameBuilder representing the item attribute "aName"
//     nameBuilder := expression.Name("aName")
type NameBuilder struct {
	name string
}

// SizeBuilder represents the output of the function size ("someName"), which
// evaluates to the size of the item attribute defined by "someName". Since
// SizeBuilder represents an operand, SizeBuilder implements the OperandBuilder
// interface. Methods and functions in the package take SizeBuilder as an
// argument and establishes relationships between operands. SizeBuilder should
// only be initialized using the function Size().
//
// Example:
//
//     // Create a SizeBuilder representing the size of the item attribute
//     // "aName"
//     sizeBuilder := expression.Name("aName").Size()
type SizeBuilder struct {
	nameBuilder NameBuilder
}

// KeyBuilder represents either the partition key or the sort key, both of which
// are top level attributes to some item in DynamoDB. Since KeyBuilder
// represents an operand, KeyBuilder implements the OperandBuilder interface.
// Methods and functions in the package take KeyBuilder as an argument and
// establishes relationships between operands. However, KeyBuilder should only
// be used to describe Key Condition Expressions. KeyBuilder should only be
// initialized using the function Key().
//
// Example:
//
//     // Create a KeyBuilder representing the item key "aKey"
//     keyBuilder := expression.Key("aKey")
type KeyBuilder struct {
	key string
}

// setValueMode specifies the type of SetValueBuilder. The default value is
// unsetValue so that an UnsetParameterError when BuildOperand() is called on an
// empty SetValueBuilder.
type setValueMode int

const (
	unsetValue setValueMode = iota
	plusValueMode
	minusValueMode
	listAppendValueMode
	ifNotExistsValueMode
)

// SetValueBuilder represents the outcome of operator functions supported by the
// DynamoDB Set operation. The operator functions are the following:
//     Plus()  // Represents the "+" operator
//     Minus() // Represents the "-" operator
//     ListAppend()
//     IfNotExists()
// For documentation on the above functions,
// see: http://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Expressions.UpdateExpressions.html#Expressions.UpdateExpressions.SET
// Since SetValueBuilder represents an operand, it implements the OperandBuilder
// interface. SetValueBuilder structs are used as arguments to the Set()
// function. SetValueBuilders should only initialize a SetValueBuilder using the
// functions listed above.
type SetValueBuilder struct {
	leftOperand  OperandBuilder
	rightOperand OperandBuilder
	mode         setValueMode
}

// Operand represents an item attribute name or value in DynamoDB. The
// relationship between Operands specified by various builders such as
// ConditionBuilders and UpdateBuilders for example is processed internally to
// write Condition Expressions and Update Expressions respectively.
type Operand struct {
	exprNode exprNode
}

// OperandBuilder represents the idea of Operand which are building blocks to
// DynamoDB Expressions. Package methods and functions can establish
// relationships between operands, representing DynamoDB Expressions. The method
// BuildOperand() is called recursively when the Build() method on the type
// Builder is called. BuildOperand() should never be called externally.
// OperandBuilder and BuildOperand() are exported to allow package functions to
// take an interface as an argument.
type OperandBuilder interface {
	BuildOperand() (Operand, error)
}

// Name creates a NameBuilder. The argument should represent the desired item
// attribute. It is possible to reference nested item attributes by using
// square brackets for lists and dots for maps. For documentation on specifying
// item attributes,
// see: http://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Expressions.Attributes.html
//
// Example:
//
//     // Specify a top-level attribute
//     name := expression.Name("TopLevel")
//     // Specify a nested attribute
//     nested := expression.Name("Record[6].SongList")
//     // Use Name() to create a condition expression
//     condition := expression.Name("foo").Equal(expression.Name("bar"))
func Name(name string) NameBuilder {
	return NameBuilder{
		name: name,
	}
}

// Value creates a ValueBuilder and sets its value to the argument. The value
// will be marshalled using the dynamodbattribute package, unless it is of
// type dynamodb.AttributeValue, where it will be used directly.
//
// Empty slices and maps will be converted to NULL dynamodb.AttributeValue
// values. If an empty value is required, pass a dynamodb.AttributeValue, e.g.:
// emptyList := (&dynamodb.AttributeValue{}).SetL([]*dynamodb.AttributeValue{})
//
// Example:
//
//     // Use Value() to create a condition expression
//     condition := expression.Name("foo").Equal(expression.Value(10))
//     // Use Value() to set the value of a set expression.
//     update := Set(expression.Name("greets"), expression.Value((&dynamodb.AttributeValue{}).SetS("hello")))
func Value(value interface{}) ValueBuilder {
	return ValueBuilder{
		value: value,
	}
}

// Size creates a SizeBuilder representing the size of the item attribute
// specified by the argument NameBuilder. Size() is only valid for certain types
// of item attributes. For documentation,
// see: http://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Expressions.OperatorsAndFunctions.html
// SizeBuilder is only a valid operand in Condition Expressions and Filter
// Expressions.
//
// Example:
//
//     // Use Size() to create a condition expression
//     condition := expression.Name("foo").Size().Equal(expression.Value(10))
//
// Expression Equivalent:
//
//     expression.Name("aName").Size()
//     "size (aName)"
func (nb NameBuilder) Size() SizeBuilder {
	return SizeBuilder{
		nameBuilder: nb,
	}
}

// Size creates a SizeBuilder representing the size of the item attribute
// specified by the argument NameBuilder. Size() is only valid for certain types
// of item attributes. For documentation,
// see: http://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Expressions.OperatorsAndFunctions.html
// SizeBuilder is only a valid operand in Condition Expressions and Filter
// Expressions.
//
// Example:
//
//     // Use Size() to create a condition expression
//     condition := expression.Size(expression.Name("foo")).Equal(expression.Value(10))
//
// Expression Equivalent:
//
//     expression.Size(expression.Name("aName"))
//     "size (aName)"
func Size(nameBuilder NameBuilder) SizeBuilder {
	return nameBuilder.Size()
}

// Key creates a KeyBuilder. The argument should represent the desired partition
// key or sort key value. KeyBuilders should only be used to specify
// relationships for Key Condition Expressions. When referring to the partition
// key or sort key in any other Expression, use Name().
//
// Example:
//
//     // Use Key() to create a key condition expression
//     keyCondition := expression.Key("foo").Equal(expression.Value("bar"))
func Key(key string) KeyBuilder {
	return KeyBuilder{
		key: key,
	}
}

// Plus creates a SetValueBuilder to be used in as an argument to Set(). The
// arguments can either be NameBuilders or ValueBuilders. Plus() only supports
// DynamoDB Number types, so the ValueBuilder must be a Number and the
// NameBuilder must specify an item attribute of type Number.
// More information: http://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Expressions.UpdateExpressions.html#Expressions.UpdateExpressions.SET.IncrementAndDecrement
//
// Example:
//
//     // Use Plus() to set the value of the item attribute "someName" to 5 + 10
//     update, err := expression.Set(expression.Name("someName"), expression.Plus(expression.Value(5), expression.Value(10)))
//
// Expression Equivalent:
//
//     expression.Plus(expression.Value(5), expression.Value(10))
//     // let :five and :ten be ExpressionAttributeValues for the values 5 and
//     // 10 respectively.
//     ":five + :ten"
func Plus(leftOperand, rightOperand OperandBuilder) SetValueBuilder {
	return SetValueBuilder{
		leftOperand:  leftOperand,
		rightOperand: rightOperand,
		mode:         plusValueMode,
	}
}

// Plus creates a SetValueBuilder to be used in as an argument to Set(). The
// arguments can either be NameBuilders or ValueBuilders. Plus() only supports
// DynamoDB Number types, so the ValueBuilder must be a Number and the
// NameBuilder must specify an item attribute of type Number.
// More information: http://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Expressions.UpdateExpressions.html#Expressions.UpdateExpressions.SET.IncrementAndDecrement
//
// Example:
//
//     // Use Plus() to set the value of the item attribute "someName" to the
//     // numeric value of item attribute "aName" incremented by 10
//     update, err := expression.Set(expression.Name("someName"), expression.Name("aName").Plus(expression.Value(10)))
//
// Expression Equivalent:
//
//     expression.Name("aName").Plus(expression.Value(10))
//     // let :ten be ExpressionAttributeValues representing the value 10
//     "aName + :ten"
func (nb NameBuilder) Plus(rightOperand OperandBuilder) SetValueBuilder {
	return Plus(nb, rightOperand)
}

// Plus creates a SetValueBuilder to be used in as an argument to Set(). The
// arguments can either be NameBuilders or ValueBuilders. Plus() only supports
// DynamoDB Number types, so the ValueBuilder must be a Number and the
// NameBuilder must specify an item attribute of type Number.
// More information: http://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Expressions.UpdateExpressions.html#Expressions.UpdateExpressions.SET.IncrementAndDecrement
//
// Example:
//
//     // Use Plus() to set the value of the item attribute "someName" to 5 + 10
//     update, err := expression.Set(expression.Name("someName"), expression.Value(5).Plus(expression.Value(10)))
//
// Expression Equivalent:
//
//     expression.Value(5).Plus(expression.Value(10))
//     // let :five and :ten be ExpressionAttributeValues representing the value
//     // 5 and 10 respectively
//     ":five + :ten"
func (vb ValueBuilder) Plus(rightOperand OperandBuilder) SetValueBuilder {
	return Plus(vb, rightOperand)
}

// Minus creates a SetValueBuilder to be used in as an argument to Set(). The
// arguments can either be NameBuilders or ValueBuilders. Minus() only supports
// DynamoDB Number types, so the ValueBuilder must be a Number and the
// NameBuilder must specify an item attribute of type Number.
// More information: http://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Expressions.UpdateExpressions.html#Expressions.UpdateExpressions.SET.IncrementAndDecrement
//
// Example:
//
//     // Use Minus() to set the value of item attribute "someName" to 5 - 10
//     update, err := expression.Set(expression.Name("someName"), expression.Minus(expression.Value(5), expression.Value(10)))
//
// Expression Equivalent:
//
//     expression.Minus(expression.Value(5), expression.Value(10))
//     // let :five and :ten be ExpressionAttributeValues for the values 5 and
//     // 10 respectively.
//     ":five - :ten"
func Minus(leftOperand, rightOperand OperandBuilder) SetValueBuilder {
	return SetValueBuilder{
		leftOperand:  leftOperand,
		rightOperand: rightOperand,
		mode:         minusValueMode,
	}
}

// Minus creates a SetValueBuilder to be used in as an argument to Set(). The
// arguments can either be NameBuilders or ValueBuilders. Minus() only supports
// DynamoDB Number types, so the ValueBuilder must be a Number and the
// NameBuilder must specify an item attribute of type Number.
// More information: http://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Expressions.UpdateExpressions.html#Expressions.UpdateExpressions.SET.IncrementAndDecrement
//
// Example:
//
//     // Use Minus() to set the value of item attribute "someName" to the
//     // numeric value of "aName" decremented by 10
//     update, err := expression.Set(expression.Name("someName"), expression.Name("aName").Minus(expression.Value(10)))
//
// Expression Equivalent:
//
//     expression.Name("aName").Minus(expression.Value(10)))
//     // let :ten be ExpressionAttributeValues represent the value 10
//     "aName - :ten"
func (nb NameBuilder) Minus(rightOperand OperandBuilder) SetValueBuilder {
	return Minus(nb, rightOperand)
}

// Minus creates a SetValueBuilder to be used in as an argument to Set(). The
// arguments can either be NameBuilders or ValueBuilders. Minus() only supports
// DynamoDB Number types, so the ValueBuilder must be a Number and the
// NameBuilder must specify an item attribute of type Number.
// More information: http://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Expressions.UpdateExpressions.html#Expressions.UpdateExpressions.SET.IncrementAndDecrement
//
// Example:
//
//     // Use Minus() to set the value of item attribute "someName" to 5 - 10
//     update, err := expression.Set(expression.Name("someName"), expression.Value(5).Minus(expression.Value(10)))
//
// Expression Equivalent:
//
//     expression.Value(5).Minus(expression.Value(10))
//     // let :five and :ten be ExpressionAttributeValues for the values 5 and
//     // 10 respectively.
//     ":five - :ten"
func (vb ValueBuilder) Minus(rightOperand OperandBuilder) SetValueBuilder {
	return Minus(vb, rightOperand)
}

// ListAppend creates a SetValueBuilder to be used in as an argument to Set().
// The arguments can either be NameBuilders or ValueBuilders. ListAppend() only
// supports DynamoDB List types, so the ValueBuilder must be a List and the
// NameBuilder must specify an item attribute of type List.
// More information: http://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Expressions.UpdateExpressions.html#Expressions.UpdateExpressions.SET.UpdatingListElements
//
// Example:
//
//     // Use ListAppend() to set item attribute "someName" to the item
//     // attribute "nameOfList" with "some" and "list" appended to it
//     update, err := expression.Set(expression.Name("someName"), expression.ListAppend(expression.Name("nameOfList"), expression.Value([]string{"some", "list"})))
//
// Expression Equivalent:
//
//     expression.ListAppend(expression.Name("nameOfList"), expression.Value([]string{"some", "list"})
//     // let :list be a ExpressionAttributeValue representing the list
//     // containing "some" and "list".
//     "list_append (nameOfList, :list)"
func ListAppend(leftOperand, rightOperand OperandBuilder) SetValueBuilder {
	return SetValueBuilder{
		leftOperand:  leftOperand,
		rightOperand: rightOperand,
		mode:         listAppendValueMode,
	}
}

// ListAppend creates a SetValueBuilder to be used in as an argument to Set().
// The arguments can either be NameBuilders or ValueBuilders. ListAppend() only
// supports DynamoDB List types, so the ValueBuilder must be a List and the
// NameBuilder must specify an item attribute of type List.
// More information: http://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Expressions.UpdateExpressions.html#Expressions.UpdateExpressions.SET.UpdatingListElements
//
// Example:
//
//     // Use ListAppend() to set item attribute "someName" to the item
//     // attribute "nameOfList" with "some" and "list" appended to it
//     update, err := expression.Set(expression.Name("someName"), expression.Name("nameOfList").ListAppend(expression.Value([]string{"some", "list"})))
//
// Expression Equivalent:
//
//     expression.Name("nameOfList").ListAppend(expression.Value([]string{"some", "list"})
//     // let :list be a ExpressionAttributeValue representing the list
//     // containing "some" and "list".
//     "list_append (nameOfList, :list)"
func (nb NameBuilder) ListAppend(rightOperand OperandBuilder) SetValueBuilder {
	return ListAppend(nb, rightOperand)
}

// ListAppend creates a SetValueBuilder to be used in as an argument to Set().
// The arguments can either be NameBuilders or ValueBuilders. ListAppend() only
// supports DynamoDB List types, so the ValueBuilder must be a List and the
// NameBuilder must specify an item attribute of type List.
// More information: http://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Expressions.UpdateExpressions.html#Expressions.UpdateExpressions.SET.UpdatingListElements
//
// Example:
//
//     // Use ListAppend() to set item attribute "someName" to a string list
//     // equal to {"a", "list", "some", "list"}
//     update, err := expression.Set(expression.Name("someName"), expression.Value([]string{"a", "list"}).ListAppend(expression.Value([]string{"some", "list"})))
//
// Expression Equivalent:
//
//     expression.Name([]string{"a", "list"}).ListAppend(expression.Value([]string{"some", "list"})
//     // let :list1 and :list2 be a ExpressionAttributeValue representing the
//     // list {"a", "list"} and {"some", "list"} respectively
//     "list_append (:list1, :list2)"
func (vb ValueBuilder) ListAppend(rightOperand OperandBuilder) SetValueBuilder {
	return ListAppend(vb, rightOperand)
}

// IfNotExists creates a SetValueBuilder to be used in as an argument to Set().
// The first argument must be a NameBuilder representing the name where the new
// item attribute is created. The second argument can either be a NameBuilder or
// a ValueBuilder. In the case that it is a NameBuilder, the value of the item
// attribute at the name specified becomes the value of the new item attribute.
// More information: http://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Expressions.UpdateExpressions.html#Expressions.UpdateExpressions.SET.PreventingAttributeOverwrites
//
// Example:
//
//     // Use IfNotExists() to set item attribute "someName" to value 5 if
//     // "someName" does not exist yet. (Prevents overwrite)
//     update, err := expression.Set(expression.Name("someName"), expression.IfNotExists(expression.Name("someName"), expression.Value(5)))
//
// Expression Equivalent:
//
//     expression.IfNotExists(expression.Name("someName"), expression.Value(5))
//     // let :five be a ExpressionAttributeValue representing the value 5
//     "if_not_exists (someName, :five)"
func IfNotExists(name NameBuilder, setValue OperandBuilder) SetValueBuilder {
	return SetValueBuilder{
		leftOperand:  name,
		rightOperand: setValue,
		mode:         ifNotExistsValueMode,
	}
}

// IfNotExists creates a SetValueBuilder to be used in as an argument to Set().
// The first argument must be a NameBuilder representing the name where the new
// item attribute is created. The second argument can either be a NameBuilder or
// a ValueBuilder. In the case that it is a NameBuilder, the value of the item
// attribute at the name specified becomes the value of the new item attribute.
// More information: http://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Expressions.UpdateExpressions.html#Expressions.UpdateExpressions.SET.PreventingAttributeOverwrites
//
// Example:
//
//     // Use IfNotExists() to set item attribute "someName" to value 5 if
//     // "someName" does not exist yet. (Prevents overwrite)
//     update, err := expression.Set(expression.Name("someName"), expression.Name("someName").IfNotExists(expression.Value(5)))
//
// Expression Equivalent:
//
//     expression.Name("someName").IfNotExists(expression.Value(5))
//     // let :five be a ExpressionAttributeValue representing the value 5
//     "if_not_exists (someName, :five)"
func (nb NameBuilder) IfNotExists(rightOperand OperandBuilder) SetValueBuilder {
	return IfNotExists(nb, rightOperand)
}

// BuildOperand creates an Operand struct which are building blocks to DynamoDB
// Expressions. Package methods and functions can establish relationships
// between operands, representing DynamoDB Expressions. The method
// BuildOperand() is called recursively when the Build() method on the type
// Builder is called. BuildOperand() should never be called externally.
// BuildOperand() aliases all strings to avoid stepping over DynamoDB's reserved
// words.
// More information on reserved words at http://docs.aws.amazon.com/amazondynamodb/latest/developerguide/ReservedWords.html
func (nb NameBuilder) BuildOperand() (Operand, error) {
	if nb.name == "" {
		return Operand{}, newUnsetParameterError("BuildOperand", "NameBuilder")
	}

	node := exprNode{
		names: []string{},
	}

	nameSplit := strings.Split(nb.name, ".")
	fmtNames := make([]string, 0, len(nameSplit))

	for _, word := range nameSplit {
		var substr string
		if word == "" {
			return Operand{}, newInvalidParameterError("BuildOperand", "NameBuilder")
		}

		if word[len(word)-1] == ']' {
			for j, char := range word {
				if char == '[' {
					substr = word[j:]
					word = word[:j]
					break
				}
			}
		}

		if word == "" {
			return Operand{}, newInvalidParameterError("BuildOperand", "NameBuilder")
		}

		// Create a string with special characters that can be substituted later: $p
		node.names = append(node.names, word)
		fmtNames = append(fmtNames, "$n"+substr)
	}
	node.fmtExpr = strings.Join(fmtNames, ".")
	return Operand{
		exprNode: node,
	}, nil
}

// BuildOperand creates an Operand struct which are building blocks to DynamoDB
// Expressions. Package methods and functions can establish relationships
// between operands, representing DynamoDB Expressions. The method
// BuildOperand() is called recursively when the Build() method on the type
// Builder is called. BuildOperand() should never be called externally.
// BuildOperand() aliases all strings to avoid stepping over DynamoDB's reserved
// words.
// More information on reserved words at http://docs.aws.amazon.com/amazondynamodb/latest/developerguide/ReservedWords.html
func (vb ValueBuilder) BuildOperand() (Operand, error) {
	var (
		expr *dynamodb.AttributeValue
		err  error
	)

	switch v := vb.value.(type) {
	case *dynamodb.AttributeValue:
		expr = v
	case dynamodb.AttributeValue:
		expr = &v
	default:
		expr, err = dynamodbattribute.Marshal(vb.value)
		if err != nil {
			return Operand{}, newInvalidParameterError("BuildOperand", "ValueBuilder")
		}
	}

	// Create a string with special characters that can be substituted later: $v
	operand := Operand{
		exprNode: exprNode{
			values:  []dynamodb.AttributeValue{*expr},
			fmtExpr: "$v",
		},
	}
	return operand, nil
}

// BuildOperand creates an Operand struct which are building blocks to DynamoDB
// Expressions. Package methods and functions can establish relationships
// between operands, representing DynamoDB Expressions. The method
// BuildOperand() is called recursively when the Build() method on the type
// Builder is called. BuildOperand() should never be called externally.
// BuildOperand() aliases all strings to avoid stepping over DynamoDB's reserved
// words.
// More information on reserved words at http://docs.aws.amazon.com/amazondynamodb/latest/developerguide/ReservedWords.html
func (sb SizeBuilder) BuildOperand() (Operand, error) {
	operand, err := sb.nameBuilder.BuildOperand()
	operand.exprNode.fmtExpr = "size (" + operand.exprNode.fmtExpr + ")"

	return operand, err
}

// BuildOperand creates an Operand struct which are building blocks to DynamoDB
// Expressions. Package methods and functions can establish relationships
// between operands, representing DynamoDB Expressions. The method
// BuildOperand() is called recursively when the Build() method on the type
// Builder is called. BuildOperand() should never be called externally.
// BuildOperand() aliases all strings to avoid stepping over DynamoDB's reserved
// words.
// More information on reserved words at http://docs.aws.amazon.com/amazondynamodb/latest/developerguide/ReservedWords.html
func (kb KeyBuilder) BuildOperand() (Operand, error) {
	if kb.key == "" {
		return Operand{}, newUnsetParameterError("BuildOperand", "KeyBuilder")
	}

	ret := Operand{
		exprNode: exprNode{
			names:   []string{kb.key},
			fmtExpr: "$n",
		},
	}

	return ret, nil
}

// BuildOperand creates an Operand struct which are building blocks to DynamoDB
// Expressions. Package methods and functions can establish relationships
// between operands, representing DynamoDB Expressions. The method
// BuildOperand() is called recursively when the Build() method on the type
// Builder is called. BuildOperand() should never be called externally.
// BuildOperand() aliases all strings to avoid stepping over DynamoDB's reserved
// words.
// More information on reserved words at http://docs.aws.amazon.com/amazondynamodb/latest/developerguide/ReservedWords.html
func (svb SetValueBuilder) BuildOperand() (Operand, error) {
	if svb.mode == unsetValue {
		return Operand{}, newUnsetParameterError("BuildOperand", "SetValueBuilder")
	}

	left, err := svb.leftOperand.BuildOperand()
	if err != nil {
		return Operand{}, err
	}
	leftNode := left.exprNode

	right, err := svb.rightOperand.BuildOperand()
	if err != nil {
		return Operand{}, err
	}
	rightNode := right.exprNode

	node := exprNode{
		children: []exprNode{leftNode, rightNode},
	}

	switch svb.mode {
	case plusValueMode:
		node.fmtExpr = "$c + $c"
	case minusValueMode:
		node.fmtExpr = "$c - $c"
	case listAppendValueMode:
		node.fmtExpr = "list_append($c, $c)"
	case ifNotExistsValueMode:
		node.fmtExpr = "if_not_exists($c, $c)"
	default:
		return Operand{}, fmt.Errorf("build operand error: unsupported mode: %v", svb.mode)
	}

	return Operand{
		exprNode: node,
	}, nil
}
