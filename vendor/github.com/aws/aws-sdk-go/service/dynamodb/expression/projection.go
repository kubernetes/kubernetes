package expression

import (
	"strings"
)

// ProjectionBuilder represents Projection Expressions in DynamoDB.
// ProjectionBuilders are the building blocks of Builders.
// More Information at: http://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Expressions.ProjectionExpressions.html
type ProjectionBuilder struct {
	names []NameBuilder
}

// NamesList returns a ProjectionBuilder representing the list of item
// attribute names specified by the argument NameBuilders. The resulting
// ProjectionBuilder can be used as a part of other ProjectionBuilders or as an
// argument to the WithProjection() method for the Builder struct.
//
// Example:
//
//     // projection represents the list of names {"foo", "bar"}
//     projection := expression.NamesList(expression.Name("foo"), expression.Name("bar"))
//
//     // Used in another Projection Expression
//     anotherProjection := expression.AddNames(projection, expression.Name("baz"))
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithProjection(newProjection)
//
// Expression Equivalent:
//
//     expression.NamesList(expression.Name("foo"), expression.Name("bar"))
//     "foo, bar"
func NamesList(nameBuilder NameBuilder, namesList ...NameBuilder) ProjectionBuilder {
	namesList = append([]NameBuilder{nameBuilder}, namesList...)
	return ProjectionBuilder{
		names: namesList,
	}
}

// NamesList returns a ProjectionBuilder representing the list of item
// attribute names specified by the argument NameBuilders. The resulting
// ProjectionBuilder can be used as a part of other ProjectionBuilders or as an
// argument to the WithProjection() method for the Builder struct.
//
// Example:
//
//     // projection represents the list of names {"foo", "bar"}
//     projection := expression.Name("foo").NamesList(expression.Name("bar"))
//
//     // Used in another Projection Expression
//     anotherProjection := expression.AddNames(projection, expression.Name("baz"))
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithProjection(newProjection)
//
// Expression Equivalent:
//
//     expression.Name("foo").NamesList(expression.Name("bar"))
//     "foo, bar"
func (nb NameBuilder) NamesList(namesList ...NameBuilder) ProjectionBuilder {
	return NamesList(nb, namesList...)
}

// AddNames returns a ProjectionBuilder representing the list of item
// attribute names equivalent to appending all of the argument item attribute
// names to the argument ProjectionBuilder. The resulting ProjectionBuilder can
// be used as a part of other ProjectionBuilders or as an argument to the
// WithProjection() method for the Builder struct.
//
// Example:
//
//     // projection represents the list of names {"foo", "bar", "baz", "qux"}
//     oldProj := expression.NamesList(expression.Name("foo"), expression.Name("bar"))
//     projection := expression.AddNames(oldProj, expression.Name("baz"), expression.Name("qux"))
//
//     // Used in another Projection Expression
//     anotherProjection := expression.AddNames(projection, expression.Name("quux"))
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithProjection(newProjection)
//
// Expression Equivalent:
//
//     expression.AddNames(expression.NamesList(expression.Name("foo"), expression.Name("bar")), expression.Name("baz"), expression.Name("qux"))
//     "foo, bar, baz, qux"
func AddNames(projectionBuilder ProjectionBuilder, namesList ...NameBuilder) ProjectionBuilder {
	projectionBuilder.names = append(projectionBuilder.names, namesList...)
	return projectionBuilder
}

// AddNames returns a ProjectionBuilder representing the list of item
// attribute names equivalent to appending all of the argument item attribute
// names to the argument ProjectionBuilder. The resulting ProjectionBuilder can
// be used as a part of other ProjectionBuilders or as an argument to the
// WithProjection() method for the Builder struct.
//
// Example:
//
//     // projection represents the list of names {"foo", "bar", "baz", "qux"}
//     oldProj := expression.NamesList(expression.Name("foo"), expression.Name("bar"))
//     projection := oldProj.AddNames(expression.Name("baz"), expression.Name("qux"))
//
//     // Used in another Projection Expression
//     anotherProjection := expression.AddNames(projection, expression.Name("quux"))
//     // Used to make an Builder
//     builder := expression.NewBuilder().WithProjection(newProjection)
//
// Expression Equivalent:
//
//     expression.NamesList(expression.Name("foo"), expression.Name("bar")).AddNames(expression.Name("baz"), expression.Name("qux"))
//     "foo, bar, baz, qux"
func (pb ProjectionBuilder) AddNames(namesList ...NameBuilder) ProjectionBuilder {
	return AddNames(pb, namesList...)
}

// buildTree builds a tree structure of exprNodes based on the tree
// structure of the input ProjectionBuilder's child NameBuilders. buildTree()
// satisfies the treeBuilder interface so ProjectionBuilder can be a part of
// Builder and Expression struct.
func (pb ProjectionBuilder) buildTree() (exprNode, error) {
	if len(pb.names) == 0 {
		return exprNode{}, newUnsetParameterError("buildTree", "ProjectionBuilder")
	}

	childNodes, err := pb.buildChildNodes()
	if err != nil {
		return exprNode{}, err
	}
	ret := exprNode{
		children: childNodes,
	}

	ret.fmtExpr = "$c" + strings.Repeat(", $c", len(pb.names)-1)

	return ret, nil
}

// buildChildNodes creates the list of the child exprNodes.
func (pb ProjectionBuilder) buildChildNodes() ([]exprNode, error) {
	childNodes := make([]exprNode, 0, len(pb.names))
	for _, name := range pb.names {
		operand, err := name.BuildOperand()
		if err != nil {
			return []exprNode{}, err
		}
		childNodes = append(childNodes, operand.exprNode)
	}

	return childNodes, nil
}
