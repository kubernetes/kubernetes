/*
Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

package build

// Walk walks the expression tree v, calling f on all subexpressions
// in a preorder traversal.
//
// The stk argument is the stack of expressions in the recursion above x,
// from outermost to innermost.
//
func Walk(v Expr, f func(x Expr, stk []Expr)) {
	var stack []Expr
	walk1(&v, &stack, func(x Expr, stk []Expr) Expr {
		f(x, stk)
		return nil
	})
}

// WalkAndUpdate walks the expression tree v, calling f on all subexpressions
// in a preorder traversal. If f returns a non-nil value, the tree is mutated.
// The new value replaces the old one.
//
// The stk argument is the stack of expressions in the recursion above x,
// from outermost to innermost.
//
func Edit(v Expr, f func(x Expr, stk []Expr) Expr) Expr {
	var stack []Expr
	return walk1(&v, &stack, f)
}

// walk1 is the actual implementation of Walk and WalkAndUpdate.
// It has the same signature and meaning as Walk,
// except that it maintains in *stack the current stack
// of nodes. Using a pointer to a slice here ensures that
// as the stack grows and shrinks the storage can be
// reused for the next growth.
func walk1(v *Expr, stack *[]Expr, f func(x Expr, stk []Expr) Expr) Expr {
	if v == nil {
		return nil
	}

	if res := f(*v, *stack); res != nil {
		*v = res
	}
	*stack = append(*stack, *v)
	switch v := (*v).(type) {
	case *File:
		for _, stmt := range v.Stmt {
			walk1(&stmt, stack, f)
		}
	case *DotExpr:
		walk1(&v.X, stack, f)
	case *IndexExpr:
		walk1(&v.X, stack, f)
		walk1(&v.Y, stack, f)
	case *KeyValueExpr:
		walk1(&v.Key, stack, f)
		walk1(&v.Value, stack, f)
	case *SliceExpr:
		walk1(&v.X, stack, f)
		if v.From != nil {
			walk1(&v.From, stack, f)
		}
		if v.To != nil {
			walk1(&v.To, stack, f)
		}
		if v.Step != nil {
			walk1(&v.Step, stack, f)
		}
	case *ParenExpr:
		walk1(&v.X, stack, f)
	case *UnaryExpr:
		walk1(&v.X, stack, f)
	case *BinaryExpr:
		walk1(&v.X, stack, f)
		walk1(&v.Y, stack, f)
	case *LambdaExpr:
		for i := range v.Var {
			walk1(&v.Var[i], stack, f)
		}
		walk1(&v.Expr, stack, f)
	case *CallExpr:
		walk1(&v.X, stack, f)
		for i := range v.List {
			walk1(&v.List[i], stack, f)
		}
	case *ListExpr:
		for i := range v.List {
			walk1(&v.List[i], stack, f)
		}
	case *SetExpr:
		for i := range v.List {
			walk1(&v.List[i], stack, f)
		}
	case *TupleExpr:
		for i := range v.List {
			walk1(&v.List[i], stack, f)
		}
	case *DictExpr:
		for i := range v.List {
			walk1(&v.List[i], stack, f)
		}
	case *ListForExpr:
		walk1(&v.X, stack, f)
		for _, c := range v.For {
			for j := range c.For.Var {
				walk1(&c.For.Var[j], stack, f)
			}
			walk1(&c.For.Expr, stack, f)
			for _, i := range c.Ifs {
				walk1(&i.Cond, stack, f)
			}
		}
	case *ConditionalExpr:
		walk1(&v.Then, stack, f)
		walk1(&v.Test, stack, f)
		walk1(&v.Else, stack, f)
	}
	*stack = (*stack)[:len(*stack)-1]
	return *v
}
