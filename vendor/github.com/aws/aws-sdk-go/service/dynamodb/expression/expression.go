package expression

import (
	"fmt"
	"sort"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/dynamodb"
)

// expressionType specifies the type of Expression. Declaring this type is used
// to eliminate magic strings
type expressionType string

const (
	projection   expressionType = "projection"
	keyCondition                = "keyCondition"
	condition                   = "condition"
	filter                      = "filter"
	update                      = "update"
)

// Implement the Sort interface
type typeList []expressionType

func (l typeList) Len() int {
	return len(l)
}

func (l typeList) Less(i, j int) bool {
	return string(l[i]) < string(l[j])
}

func (l typeList) Swap(i, j int) {
	l[i], l[j] = l[j], l[i]
}

// Builder represents the struct that builds the Expression struct. Methods such
// as WithProjection() and WithCondition() can add different kinds of DynamoDB
// Expressions to the Builder. The method Build() creates an Expression struct
// with the specified types of DynamoDB Expressions.
//
// Example:
//
//     keyCond := expression.Key("someKey").Equal(expression.Value("someValue"))
//     proj := expression.NamesList(expression.Name("aName"), expression.Name("anotherName"), expression.Name("oneOtherName"))
//
//     builder := expression.NewBuilder().WithKeyCondition(keyCond).WithProjection(proj)
//     expression := builder.Build()
//
//     queryInput := dynamodb.QueryInput{
//       KeyConditionExpression:    expression.KeyCondition(),
//       ProjectionExpression:      expression.Projection(),
//       ExpressionAttributeNames:  expression.Names(),
//       ExpressionAttributeValues: expression.Values(),
//       TableName: aws.String("SomeTable"),
//     }
type Builder struct {
	expressionMap map[expressionType]treeBuilder
}

// NewBuilder returns an empty Builder struct. Methods such as WithProjection()
// and WithCondition() can add different kinds of DynamoDB Expressions to the
// Builder. The method Build() creates an Expression struct with the specified
// types of DynamoDB Expressions.
//
// Example:
//
//     keyCond := expression.Key("someKey").Equal(expression.Value("someValue"))
//     proj := expression.NamesList(expression.Name("aName"), expression.Name("anotherName"), expression.Name("oneOtherName"))
//     builder := expression.NewBuilder().WithKeyCondition(keyCond).WithProjection(proj)
func NewBuilder() Builder {
	return Builder{}
}

// Build builds an Expression struct representing multiple types of DynamoDB
// Expressions. Getter methods on the resulting Expression struct returns the
// DynamoDB Expression strings as well as the maps that correspond to
// ExpressionAttributeNames and ExpressionAttributeValues. Calling Build() on an
// empty Builder returns the typed error EmptyParameterError.
//
// Example:
//
//     // keyCond represents the Key Condition Expression
//     keyCond := expression.Key("someKey").Equal(expression.Value("someValue"))
//     // proj represents the Projection Expression
//     proj := expression.NamesList(expression.Name("aName"), expression.Name("anotherName"), expression.Name("oneOtherName"))
//
//     // Add keyCond and proj to builder as a Key Condition and Projection
//     // respectively
//     builder := expression.NewBuilder().WithKeyCondition(keyCond).WithProjection(proj)
//     expression := builder.Build()
//
//     queryInput := dynamodb.QueryInput{
//       KeyConditionExpression:    expression.KeyCondition(),
//       ProjectionExpression:      expression.Projection(),
//       ExpressionAttributeNames:  expression.Names(),
//       ExpressionAttributeValues: expression.Values(),
//       TableName: aws.String("SomeTable"),
//     }
func (b Builder) Build() (Expression, error) {
	if b.expressionMap == nil {
		return Expression{}, newUnsetParameterError("Build", "Builder")
	}

	aliasList, expressionMap, err := b.buildChildTrees()
	if err != nil {
		return Expression{}, err
	}

	expression := Expression{
		expressionMap: expressionMap,
	}

	if len(aliasList.namesList) != 0 {
		namesMap := map[string]*string{}
		for ind, val := range aliasList.namesList {
			namesMap[fmt.Sprintf("#%v", ind)] = aws.String(val)
		}
		expression.namesMap = namesMap
	}

	if len(aliasList.valuesList) != 0 {
		valuesMap := map[string]*dynamodb.AttributeValue{}
		for i := 0; i < len(aliasList.valuesList); i++ {
			valuesMap[fmt.Sprintf(":%v", i)] = &aliasList.valuesList[i]
		}
		expression.valuesMap = valuesMap
	}

	return expression, nil
}

// buildChildTrees compiles the list of treeBuilders that are the children of
// the argument Builder. The returned aliasList represents all the alias tokens
// used in the expression strings. The returned map[string]string maps the type
// of expression (i.e. "condition", "update") to the appropriate expression
// string.
func (b Builder) buildChildTrees() (aliasList, map[expressionType]string, error) {
	aList := aliasList{}
	formattedExpressions := map[expressionType]string{}
	keys := typeList{}

	for expressionType := range b.expressionMap {
		keys = append(keys, expressionType)
	}

	sort.Sort(keys)

	for _, key := range keys {
		node, err := b.expressionMap[key].buildTree()
		if err != nil {
			return aliasList{}, nil, err
		}
		formattedExpression, err := node.buildExpressionString(&aList)
		if err != nil {
			return aliasList{}, nil, err
		}
		formattedExpressions[key] = formattedExpression
	}

	return aList, formattedExpressions, nil
}

// WithCondition method adds the argument ConditionBuilder as a Condition
// Expression to the argument Builder. If the argument Builder already has a
// ConditionBuilder representing a Condition Expression, WithCondition()
// overwrites the existing ConditionBuilder.
//
// Example:
//
//     // let builder be an existing Builder{} and cond be an existing
//     // ConditionBuilder{}
//     builder = builder.WithCondition(cond)
//
//     // add other DynamoDB Expressions to the builder. let proj be an already
//     // existing ProjectionBuilder
//     builder = builder.WithProjection(proj)
//     // create an Expression struct
//     expression := builder.Build()
func (b Builder) WithCondition(conditionBuilder ConditionBuilder) Builder {
	if b.expressionMap == nil {
		b.expressionMap = map[expressionType]treeBuilder{}
	}
	b.expressionMap[condition] = conditionBuilder
	return b
}

// WithProjection method adds the argument ProjectionBuilder as a Projection
// Expression to the argument Builder. If the argument Builder already has a
// ProjectionBuilder representing a Projection Expression, WithProjection()
// overwrites the existing ProjectionBuilder.
//
// Example:
//
//     // let builder be an existing Builder{} and proj be an existing
//     // ProjectionBuilder{}
//     builder = builder.WithProjection(proj)
//
//     // add other DynamoDB Expressions to the builder. let cond be an already
//     // existing ConditionBuilder
//     builder = builder.WithCondition(cond)
//     // create an Expression struct
//     expression := builder.Build()
func (b Builder) WithProjection(projectionBuilder ProjectionBuilder) Builder {
	if b.expressionMap == nil {
		b.expressionMap = map[expressionType]treeBuilder{}
	}
	b.expressionMap[projection] = projectionBuilder
	return b
}

// WithKeyCondition method adds the argument KeyConditionBuilder as a Key
// Condition Expression to the argument Builder. If the argument Builder already
// has a KeyConditionBuilder representing a Key Condition Expression,
// WithKeyCondition() overwrites the existing KeyConditionBuilder.
//
// Example:
//
//     // let builder be an existing Builder{} and keyCond be an existing
//     // KeyConditionBuilder{}
//     builder = builder.WithKeyCondition(keyCond)
//
//     // add other DynamoDB Expressions to the builder. let cond be an already
//     // existing ConditionBuilder
//     builder = builder.WithCondition(cond)
//     // create an Expression struct
//     expression := builder.Build()
func (b Builder) WithKeyCondition(keyConditionBuilder KeyConditionBuilder) Builder {
	if b.expressionMap == nil {
		b.expressionMap = map[expressionType]treeBuilder{}
	}
	b.expressionMap[keyCondition] = keyConditionBuilder
	return b
}

// WithFilter method adds the argument ConditionBuilder as a Filter Expression
// to the argument Builder. If the argument Builder already has a
// ConditionBuilder representing a Filter Expression, WithFilter()
// overwrites the existing ConditionBuilder.
//
// Example:
//
//     // let builder be an existing Builder{} and filt be an existing
//     // ConditionBuilder{}
//     builder = builder.WithFilter(filt)
//
//     // add other DynamoDB Expressions to the builder. let cond be an already
//     // existing ConditionBuilder
//     builder = builder.WithCondition(cond)
//     // create an Expression struct
//     expression := builder.Build()
func (b Builder) WithFilter(filterBuilder ConditionBuilder) Builder {
	if b.expressionMap == nil {
		b.expressionMap = map[expressionType]treeBuilder{}
	}
	b.expressionMap[filter] = filterBuilder
	return b
}

// WithUpdate method adds the argument UpdateBuilder as an Update Expression
// to the argument Builder. If the argument Builder already has a UpdateBuilder
// representing a Update Expression, WithUpdate() overwrites the existing
// UpdateBuilder.
//
// Example:
//
//     // let builder be an existing Builder{} and update be an existing
//     // UpdateBuilder{}
//     builder = builder.WithUpdate(update)
//
//     // add other DynamoDB Expressions to the builder. let cond be an already
//     // existing ConditionBuilder
//     builder = builder.WithCondition(cond)
//     // create an Expression struct
//     expression := builder.Build()
func (b Builder) WithUpdate(updateBuilder UpdateBuilder) Builder {
	if b.expressionMap == nil {
		b.expressionMap = map[expressionType]treeBuilder{}
	}
	b.expressionMap[update] = updateBuilder
	return b
}

// Expression represents a collection of DynamoDB Expressions. The getter
// methods of the Expression struct retrieves the formatted DynamoDB
// Expressions, ExpressionAttributeNames, and ExpressionAttributeValues.
//
// Example:
//
//     // keyCond represents the Key Condition Expression
//     keyCond := expression.Key("someKey").Equal(expression.Value("someValue"))
//     // proj represents the Projection Expression
//     proj := expression.NamesList(expression.Name("aName"), expression.Name("anotherName"), expression.Name("oneOtherName"))
//
//     // Add keyCond and proj to builder as a Key Condition and Projection
//     // respectively
//     builder := expression.NewBuilder().WithKeyCondition(keyCond).WithProjection(proj)
//     expression := builder.Build()
//
//     queryInput := dynamodb.QueryInput{
//       KeyConditionExpression:    expression.KeyCondition(),
//       ProjectionExpression:      expression.Projection(),
//       ExpressionAttributeNames:  expression.Names(),
//       ExpressionAttributeValues: expression.Values(),
//       TableName: aws.String("SomeTable"),
//     }
type Expression struct {
	expressionMap map[expressionType]string
	namesMap      map[string]*string
	valuesMap     map[string]*dynamodb.AttributeValue
}

// treeBuilder interface is fulfilled by builder structs that represent
// different types of Expressions.
type treeBuilder interface {
	// buildTree creates the tree structure of exprNodes. The tree structure
	// of exprNodes are traversed in order to build the string representing
	// different types of Expressions as well as the maps that represent
	// ExpressionAttributeNames and ExpressionAttributeValues.
	buildTree() (exprNode, error)
}

// Condition returns the *string corresponding to the Condition Expression
// of the argument Expression. This method is used to satisfy the members of
// DynamoDB input structs. If the Expression does not have a condition
// expression this method returns nil.
//
// Example:
//
//     // let expression be an instance of Expression{}
//
//     deleteInput := dynamodb.DeleteItemInput{
//       ConditionExpression:       expression.Condition(),
//       ExpressionAttributeNames:  expression.Names(),
//       ExpressionAttributeValues: expression.Values(),
//       Key: map[string]*dynamodb.AttributeValue{
//         "PartitionKey": &dynamodb.AttributeValue{
//           S: aws.String("SomeKey"),
//         },
//       },
//       TableName: aws.String("SomeTable"),
//     }
func (e Expression) Condition() *string {
	return e.returnExpression(condition)
}

// Filter returns the *string corresponding to the Filter Expression of the
// argument Expression. This method is used to satisfy the members of DynamoDB
// input structs. If the Expression does not have a filter expression this
// method returns nil.
//
// Example:
//
//     // let expression be an instance of Expression{}
//
//     queryInput := dynamodb.QueryInput{
//       KeyConditionExpression:    expression.KeyCondition(),
//       FilterExpression:          expression.Filter(),
//       ExpressionAttributeNames:  expression.Names(),
//       ExpressionAttributeValues: expression.Values(),
//       TableName: aws.String("SomeTable"),
//     }
func (e Expression) Filter() *string {
	return e.returnExpression(filter)
}

// Projection returns the *string corresponding to the Projection Expression
// of the argument Expression. This method is used to satisfy the members of
// DynamoDB input structs. If the Expression does not have a projection
// expression this method returns nil.
//
// Example:
//
//     // let expression be an instance of Expression{}
//
//     queryInput := dynamodb.QueryInput{
//       KeyConditionExpression:    expression.KeyCondition(),
//       ProjectionExpression:      expression.Projection(),
//       ExpressionAttributeNames:  expression.Names(),
//       ExpressionAttributeValues: expression.Values(),
//       TableName: aws.String("SomeTable"),
//     }
func (e Expression) Projection() *string {
	return e.returnExpression(projection)
}

// KeyCondition returns the *string corresponding to the Key Condition
// Expression of the argument Expression. This method is used to satisfy the
// members of DynamoDB input structs. If the argument Expression does not have a
// KeyConditionExpression, KeyCondition() returns nil.
//
// Example:
//
//     // let expression be an instance of Expression{}
//
//     queryInput := dynamodb.QueryInput{
//       KeyConditionExpression:    expression.KeyCondition(),
//       ProjectionExpression:      expression.Projection(),
//       ExpressionAttributeNames:  expression.Names(),
//       ExpressionAttributeValues: expression.Values(),
//       TableName: aws.String("SomeTable"),
//     }
func (e Expression) KeyCondition() *string {
	return e.returnExpression(keyCondition)
}

// Update returns the *string corresponding to the Update Expression of the
// argument Expression. This method is used to satisfy the members of DynamoDB
// input structs. If the argument Expression does not have a UpdateExpression,
// Update() returns nil.
//
// Example:
//
//     // let expression be an instance of Expression{}
//
//     updateInput := dynamodb.UpdateInput{
//       Key: map[string]*dynamodb.AttributeValue{
//         "PartitionKey": {
//           S: aws.String("someKey"),
//         },
//       },
//       UpdateExpression:          expression.Update(),
//       ExpressionAttributeNames:  expression.Names(),
//       ExpressionAttributeValues: expression.Values(),
//       TableName: aws.String("SomeTable"),
//     }
func (e Expression) Update() *string {
	return e.returnExpression(update)
}

// Names returns the map[string]*string corresponding to the
// ExpressionAttributeNames of the argument Expression. This method is used to
// satisfy the members of DynamoDB input structs. If Expression does not use
// ExpressionAttributeNames, this method returns nil. The
// ExpressionAttributeNames and ExpressionAttributeValues member of the input
// struct must always be assigned when using the Expression struct since all
// item attribute names and values are aliased. That means that if the
// ExpressionAttributeNames and ExpressionAttributeValues member is not assigned
// with the corresponding Names() and Values() methods, the DynamoDB operation
// will run into a logic error.
//
// Example:
//
//     // let expression be an instance of Expression{}
//
//     queryInput := dynamodb.QueryInput{
//       KeyConditionExpression:    expression.KeyCondition(),
//       ProjectionExpression:      expression.Projection(),
//       ExpressionAttributeNames:  expression.Names(),
//       ExpressionAttributeValues: expression.Values(),
//       TableName: aws.String("SomeTable"),
//     }
func (e Expression) Names() map[string]*string {
	return e.namesMap
}

// Values returns the map[string]*dynamodb.AttributeValue corresponding to
// the ExpressionAttributeValues of the argument Expression. This method is used
// to satisfy the members of DynamoDB input structs. If Expression does not use
// ExpressionAttributeValues, this method returns nil. The
// ExpressionAttributeNames and ExpressionAttributeValues member of the input
// struct must always be assigned when using the Expression struct since all
// item attribute names and values are aliased. That means that if the
// ExpressionAttributeNames and ExpressionAttributeValues member is not assigned
// with the corresponding Names() and Values() methods, the DynamoDB operation
// will run into a logic error.
//
// Example:
//
//     // let expression be an instance of Expression{}
//
//     queryInput := dynamodb.QueryInput{
//       KeyConditionExpression:    expression.KeyCondition(),
//       ProjectionExpression:      expression.Projection(),
//       ExpressionAttributeNames:  expression.Names(),
//       ExpressionAttributeValues: expression.Values(),
//       TableName: aws.String("SomeTable"),
//     }
func (e Expression) Values() map[string]*dynamodb.AttributeValue {
	return e.valuesMap
}

// returnExpression returns *string corresponding to the type of Expression
// string specified by the expressionType. If there is no corresponding
// expression available in Expression, the method returns nil
func (e Expression) returnExpression(expressionType expressionType) *string {
	if e.expressionMap == nil {
		return nil
	}
	return aws.String(e.expressionMap[expressionType])
}

// exprNode are the generic nodes that represents both Operands and
// Conditions. The purpose of exprNode is to be able to call an generic
// recursive function on the top level exprNode to be able to determine a root
// node in order to deduplicate name aliases.
// fmtExpr is a string that has escaped characters to refer to
// names/values/children which needs to be aliased at runtime in order to avoid
// duplicate values. The rules are as follows:
//     $n: Indicates that an alias of a name needs to be inserted. The
//         corresponding name to be alias is in the []names slice.
//     $v: Indicates that an alias of a value needs to be inserted. The
//         corresponding value to be alias is in the []values slice.
//     $c: Indicates that the fmtExpr of a child exprNode needs to be inserted.
//         The corresponding child node is in the []children slice.
type exprNode struct {
	names    []string
	values   []dynamodb.AttributeValue
	children []exprNode
	fmtExpr  string
}

// aliasList keeps track of all the names we need to alias in the nested
// struct of conditions and operands. This allows each alias to be unique.
// aliasList is passed in as a pointer when buildChildTrees is called in
// order to deduplicate all names within the tree strcuture of the exprNodes.
type aliasList struct {
	namesList  []string
	valuesList []dynamodb.AttributeValue
}

// buildExpressionString returns a string with aliasing for names/values
// specified by aliasList. The string corresponds to the expression that the
// exprNode tree represents.
func (en exprNode) buildExpressionString(aliasList *aliasList) (string, error) {
	// Since each exprNode contains a slice of names, values, and children that
	// correspond to the escaped characters, we an index to traverse the slices
	index := struct {
		name, value, children int
	}{}

	formattedExpression := en.fmtExpr

	for i := 0; i < len(formattedExpression); {
		if formattedExpression[i] != '$' {
			i++
			continue
		}

		if i == len(formattedExpression)-1 {
			return "", fmt.Errorf("buildexprNode error: invalid escape character")
		}

		var alias string
		var err error
		// if an escaped character is found, substitute it with the proper alias
		// TODO consider AST instead of string in the future
		switch formattedExpression[i+1] {
		case 'n':
			alias, err = substitutePath(index.name, en, aliasList)
			if err != nil {
				return "", err
			}
			index.name++

		case 'v':
			alias, err = substituteValue(index.value, en, aliasList)
			if err != nil {
				return "", err
			}
			index.value++

		case 'c':
			alias, err = substituteChild(index.children, en, aliasList)
			if err != nil {
				return "", err
			}
			index.children++

		default:
			return "", fmt.Errorf("buildexprNode error: invalid escape rune %#v", formattedExpression[i+1])
		}
		formattedExpression = formattedExpression[:i] + alias + formattedExpression[i+2:]
		i += len(alias)
	}

	return formattedExpression, nil
}

// substitutePath substitutes the escaped character $n with the appropriate
// alias.
func substitutePath(index int, node exprNode, aliasList *aliasList) (string, error) {
	if index >= len(node.names) {
		return "", fmt.Errorf("substitutePath error: exprNode []names out of range")
	}
	str, err := aliasList.aliasPath(node.names[index])
	if err != nil {
		return "", err
	}
	return str, nil
}

// substituteValue substitutes the escaped character $v with the appropriate
// alias.
func substituteValue(index int, node exprNode, aliasList *aliasList) (string, error) {
	if index >= len(node.values) {
		return "", fmt.Errorf("substituteValue error: exprNode []values out of range")
	}
	str, err := aliasList.aliasValue(node.values[index])
	if err != nil {
		return "", err
	}
	return str, nil
}

// substituteChild substitutes the escaped character $c with the appropriate
// alias.
func substituteChild(index int, node exprNode, aliasList *aliasList) (string, error) {
	if index >= len(node.children) {
		return "", fmt.Errorf("substituteChild error: exprNode []children out of range")
	}
	return node.children[index].buildExpressionString(aliasList)
}

// aliasValue returns the corresponding alias to the dav value argument. Since
// values are not deduplicated as of now, all values are just appended to the
// aliasList and given the index as the alias.
func (al *aliasList) aliasValue(dav dynamodb.AttributeValue) (string, error) {
	al.valuesList = append(al.valuesList, dav)
	return fmt.Sprintf(":%d", len(al.valuesList)-1), nil
}

// aliasPath returns the corresponding alias to the argument string. The
// argument is checked against all existing aliasList names in order to avoid
// duplicate strings getting two different aliases.
func (al *aliasList) aliasPath(nm string) (string, error) {
	for ind, name := range al.namesList {
		if nm == name {
			return fmt.Sprintf("#%d", ind), nil
		}
	}
	al.namesList = append(al.namesList, nm)
	return fmt.Sprintf("#%d", len(al.namesList)-1), nil
}
