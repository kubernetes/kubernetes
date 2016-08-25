package parser

import "github.com/hashicorp/hcl/hcl/ast"

// flattenObjects takes an AST node, walks it, and flattens
func flattenObjects(node ast.Node) {
	ast.Walk(node, func(n ast.Node) (ast.Node, bool) {
		// We only care about lists, because this is what we modify
		list, ok := n.(*ast.ObjectList)
		if !ok {
			return n, true
		}

		// Rebuild the item list
		items := make([]*ast.ObjectItem, 0, len(list.Items))
		frontier := make([]*ast.ObjectItem, len(list.Items))
		copy(frontier, list.Items)
		for len(frontier) > 0 {
			// Pop the current item
			n := len(frontier)
			item := frontier[n-1]
			frontier = frontier[:n-1]

			switch v := item.Val.(type) {
			case *ast.ObjectType:
				items, frontier = flattenObjectType(v, item, items, frontier)
			case *ast.ListType:
				items, frontier = flattenListType(v, item, items, frontier)
			default:
				items = append(items, item)
			}
		}

		// Reverse the list since the frontier model runs things backwards
		for i := len(items)/2 - 1; i >= 0; i-- {
			opp := len(items) - 1 - i
			items[i], items[opp] = items[opp], items[i]
		}

		// Done! Set the original items
		list.Items = items
		return n, true
	})
}

func flattenListType(
	ot *ast.ListType,
	item *ast.ObjectItem,
	items []*ast.ObjectItem,
	frontier []*ast.ObjectItem) ([]*ast.ObjectItem, []*ast.ObjectItem) {
	// All the elements of this object must also be objects!
	for _, subitem := range ot.List {
		if _, ok := subitem.(*ast.ObjectType); !ok {
			items = append(items, item)
			return items, frontier
		}
	}

	// Great! We have a match go through all the items and flatten
	for _, elem := range ot.List {
		// Add it to the frontier so that we can recurse
		frontier = append(frontier, &ast.ObjectItem{
			Keys:        item.Keys,
			Assign:      item.Assign,
			Val:         elem,
			LeadComment: item.LeadComment,
			LineComment: item.LineComment,
		})
	}

	return items, frontier
}

func flattenObjectType(
	ot *ast.ObjectType,
	item *ast.ObjectItem,
	items []*ast.ObjectItem,
	frontier []*ast.ObjectItem) ([]*ast.ObjectItem, []*ast.ObjectItem) {
	// If the list has no items we do not have to flatten anything
	if ot.List.Items == nil {
		items = append(items, item)
		return items, frontier
	}

	// All the elements of this object must also be objects!
	for _, subitem := range ot.List.Items {
		if _, ok := subitem.Val.(*ast.ObjectType); !ok {
			items = append(items, item)
			return items, frontier
		}
	}

	// Great! We have a match go through all the items and flatten
	for _, subitem := range ot.List.Items {
		// Copy the new key
		keys := make([]*ast.ObjectKey, len(item.Keys)+len(subitem.Keys))
		copy(keys, item.Keys)
		copy(keys[len(item.Keys):], subitem.Keys)

		// Add it to the frontier so that we can recurse
		frontier = append(frontier, &ast.ObjectItem{
			Keys:        keys,
			Assign:      item.Assign,
			Val:         subitem.Val,
			LeadComment: item.LeadComment,
			LineComment: item.LineComment,
		})
	}

	return items, frontier
}
