// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package kio

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/xlab/treeprint"
	"sigs.k8s.io/kustomize/kyaml/kio/kioutil"
	"sigs.k8s.io/kustomize/kyaml/yaml"
)

type TreeStructure string

const (
	// TreeStructurePackage configures TreeWriter to generate the tree structure off of the
	// Resources packages.
	TreeStructurePackage TreeStructure = "directory"

	// TreeStructureOwners configures TreeWriter to generate the tree structure off of the
	// Resource owners.
	TreeStructureGraph TreeStructure = "owners"
)

var GraphStructures = []string{string(TreeStructureGraph), string(TreeStructurePackage)}

// TreeWriter prints the package structured as a tree.
// TODO(pwittrock): test this package better.  it is lower-risk since it is only
// used for printing rather than updating or editing.
type TreeWriter struct {
	Writer          io.Writer
	Root            string
	Fields          []TreeWriterField
	Structure       TreeStructure
	OpenAPIFileName string
}

// TreeWriterField configures a Resource field to be included in the tree
type TreeWriterField struct {
	yaml.PathMatcher
	Name    string
	SubName string
}

func (p TreeWriter) packageStructure(nodes []*yaml.RNode) error {
	indexByPackage := p.index(nodes)

	// create the new tree
	tree := treeprint.New()
	tree.SetValue(p.Root)

	// add each package to the tree
	treeIndex := map[string]treeprint.Tree{}
	keys := p.sort(indexByPackage)
	for _, pkg := range keys {
		// create a branch for this package -- search for the parent package and create
		// the branch under it -- requires that the keys are sorted
		branch := tree
		for parent, subTree := range treeIndex {
			if strings.HasPrefix(pkg, parent) {
				// found a package whose path is a prefix to our own, use this
				// package if a closer one isn't found
				branch = subTree
				// don't break, continue searching for more closely related ancestors
			}
		}

		// create a new branch for the package
		createOk := pkg != "." // special edge case logic for tree on current working dir
		if createOk {
			branch = branch.AddBranch(branchName(p.Root, pkg, p.OpenAPIFileName))
		}

		// cache the branch for this package
		treeIndex[pkg] = branch

		// print each resource in the package
		for i := range indexByPackage[pkg] {
			var err error
			if _, err = p.doResource(indexByPackage[pkg][i], "", branch); err != nil {
				return err
			}
		}
	}

	_, err := io.WriteString(p.Writer, tree.String())
	return err
}

// branchName takes the root directory and relative path to the directory
// and returns the branch name
func branchName(root, dirRelPath, openAPIFileName string) string {
	name := filepath.Base(dirRelPath)
	_, err := os.Stat(filepath.Join(root, dirRelPath, openAPIFileName))
	if !os.IsNotExist(err) {
		// add Pkg: prefix indicating that it is a separate package as it has
		// openAPIFile
		return fmt.Sprintf("Pkg: %s", name)
	}
	return name
}

// Write writes the ascii tree to p.Writer
func (p TreeWriter) Write(nodes []*yaml.RNode) error {
	switch p.Structure {
	case TreeStructurePackage:
		return p.packageStructure(nodes)
	case TreeStructureGraph:
		return p.graphStructure(nodes)
	}

	// If any resource has an owner reference, default to the graph structure. Otherwise, use package structure.
	for _, node := range nodes {
		if owners, _ := node.Pipe(yaml.Lookup("metadata", "ownerReferences")); owners != nil {
			return p.graphStructure(nodes)
		}
	}
	return p.packageStructure(nodes)
}

// node wraps a tree node, and any children nodes
type node struct {
	p TreeWriter
	*yaml.RNode
	children []*node
}

func (a node) Len() int      { return len(a.children) }
func (a node) Swap(i, j int) { a.children[i], a.children[j] = a.children[j], a.children[i] }
func (a node) Less(i, j int) bool {
	return compareNodes(a.children[i].RNode, a.children[j].RNode)
}

// Tree adds this node to the root
func (a node) Tree(root treeprint.Tree) error {
	sort.Sort(a)
	branch := root
	var err error

	// generate a node for the Resource
	if a.RNode != nil {
		branch, err = a.p.doResource(a.RNode, "Resource", root)
		if err != nil {
			return err
		}
	}

	// attach children to the branch
	for _, n := range a.children {
		if err := n.Tree(branch); err != nil {
			return err
		}
	}
	return nil
}

// graphStructure writes the tree using owners for structure
func (p TreeWriter) graphStructure(nodes []*yaml.RNode) error {
	resourceToOwner := map[string]*node{}
	root := &node{}
	// index each of the nodes by their owner
	for _, n := range nodes {
		ownerVal, err := ownerToString(n)
		if err != nil {
			return err
		}
		var owner *node
		if ownerVal == "" {
			// no owner -- attach to the root
			owner = root
		} else {
			// owner found -- attach to the owner
			var found bool
			owner, found = resourceToOwner[ownerVal]
			if !found {
				// initialize the owner if not found
				resourceToOwner[ownerVal] = &node{p: p}
				owner = resourceToOwner[ownerVal]
			}
		}

		nodeVal, err := nodeToString(n)
		if err != nil {
			return err
		}
		val, found := resourceToOwner[nodeVal]
		if !found {
			// initialize the node if not found -- may have already been initialized if it
			// is the owner of another node
			resourceToOwner[nodeVal] = &node{p: p}
			val = resourceToOwner[nodeVal]
		}
		val.RNode = n
		owner.children = append(owner.children, val)
	}

	for k, v := range resourceToOwner {
		if v.RNode == nil {
			return fmt.Errorf(
				"owner '%s' not found in input, but found as an owner of input objects", k)
		}
	}

	// print the tree
	tree := treeprint.New()
	if err := root.Tree(tree); err != nil {
		return err
	}

	_, err := io.WriteString(p.Writer, tree.String())
	return err
}

// nodeToString generates a string to identify the node -- matches ownerToString format
func nodeToString(node *yaml.RNode) (string, error) {
	meta, err := node.GetMeta()
	if err != nil {
		return "", err
	}

	return fmt.Sprintf("%s %s/%s", meta.Kind, meta.Namespace, meta.Name), nil
}

// ownerToString generate a string to identify the owner -- matches nodeToString format
func ownerToString(node *yaml.RNode) (string, error) {
	meta, err := node.GetMeta()
	if err != nil {
		return "", err
	}
	namespace := meta.Namespace

	owners, err := node.Pipe(yaml.Lookup("metadata", "ownerReferences"))
	if err != nil {
		return "", err
	}
	if owners == nil {
		return "", nil
	}

	elements, err := owners.Elements()
	if err != nil {
		return "", err
	}
	if len(elements) == 0 {
		return "", err
	}
	owner := elements[0]
	var kind, name string

	if value := owner.Field("kind"); !value.IsNilOrEmpty() {
		kind = value.Value.YNode().Value
	}
	if value := owner.Field("name"); !value.IsNilOrEmpty() {
		name = value.Value.YNode().Value
	}

	return fmt.Sprintf("%s %s/%s", kind, namespace, name), nil
}

// index indexes the Resources by their package
func (p TreeWriter) index(nodes []*yaml.RNode) map[string][]*yaml.RNode {
	// index the ResourceNodes by package
	indexByPackage := map[string][]*yaml.RNode{}
	for i := range nodes {
		meta, err := nodes[i].GetMeta()
		if err != nil || meta.Kind == "" {
			// not a resource
			continue
		}
		pkg := filepath.Dir(meta.Annotations[kioutil.PathAnnotation])
		indexByPackage[pkg] = append(indexByPackage[pkg], nodes[i])
	}
	return indexByPackage
}

func compareNodes(i, j *yaml.RNode) bool {
	metai, _ := i.GetMeta()
	metaj, _ := j.GetMeta()
	pi := metai.Annotations[kioutil.PathAnnotation]
	pj := metaj.Annotations[kioutil.PathAnnotation]

	// compare file names
	if filepath.Base(pi) != filepath.Base(pj) {
		return filepath.Base(pi) < filepath.Base(pj)
	}

	// compare namespace
	if metai.Namespace != metaj.Namespace {
		return metai.Namespace < metaj.Namespace
	}

	// compare name
	if metai.Name != metaj.Name {
		return metai.Name < metaj.Name
	}

	// compare kind
	if metai.Kind != metaj.Kind {
		return metai.Kind < metaj.Kind
	}

	// compare apiVersion
	if metai.APIVersion != metaj.APIVersion {
		return metai.APIVersion < metaj.APIVersion
	}
	return true
}

// sort sorts the Resources in the index in display order and returns the ordered
// keys for the index
//
// Packages are sorted by package name
// Resources within a package are sorted by: [filename, namespace, name, kind, apiVersion]
func (p TreeWriter) sort(indexByPackage map[string][]*yaml.RNode) []string {
	var keys []string
	for k := range indexByPackage {
		pkgNodes := indexByPackage[k]
		sort.Slice(pkgNodes, func(i, j int) bool { return compareNodes(pkgNodes[i], pkgNodes[j]) })
		keys = append(keys, k)
	}

	// return the package names sorted lexicographically
	sort.Strings(keys)
	return keys
}

func (p TreeWriter) doResource(leaf *yaml.RNode, metaString string, branch treeprint.Tree) (treeprint.Tree, error) {
	meta, _ := leaf.GetMeta()
	if metaString == "" {
		path := meta.Annotations[kioutil.PathAnnotation]
		path = filepath.Base(path)
		metaString = path
	}

	value := fmt.Sprintf("%s %s", meta.Kind, meta.Name)
	if len(meta.Namespace) > 0 {
		value = fmt.Sprintf("%s %s/%s", meta.Kind, meta.Namespace, meta.Name)
	}

	fields, err := p.getFields(leaf)
	if err != nil {
		return nil, err
	}

	n := branch.AddMetaBranch(metaString, value)
	for i := range fields {
		field := fields[i]

		// do leaf node
		if len(field.matchingElementsAndFields) == 0 {
			n.AddNode(fmt.Sprintf("%s: %s", field.name, field.value))
			continue
		}

		// do nested nodes
		b := n.AddBranch(field.name)
		for j := range field.matchingElementsAndFields {
			elem := field.matchingElementsAndFields[j]
			b := b.AddBranch(elem.name)
			for k := range elem.matchingElementsAndFields {
				field := elem.matchingElementsAndFields[k]
				b.AddNode(fmt.Sprintf("%s: %s", field.name, field.value))
			}
		}
	}

	return n, nil
}

// getFields looks up p.Fields from leaf and structures them into treeFields.
// TODO(pwittrock): simplify this function
func (p TreeWriter) getFields(leaf *yaml.RNode) (treeFields, error) {
	fieldsByName := map[string]*treeField{}

	// index nested and non-nested fields
	for i := range p.Fields {
		f := p.Fields[i]
		seq, err := leaf.Pipe(&f)
		if err != nil {
			return nil, err
		}
		if seq == nil {
			continue
		}

		if fieldsByName[f.Name] == nil {
			fieldsByName[f.Name] = &treeField{name: f.Name}
		}

		// non-nested field -- add directly to the treeFields list
		if f.SubName == "" {
			// non-nested field -- only 1 element
			val, err := yaml.String(seq.Content()[0], yaml.Trim, yaml.Flow)
			if err != nil {
				return nil, err
			}
			fieldsByName[f.Name].value = val
			continue
		}

		// nested-field -- create a parent elem, and index by the 'match' value
		if fieldsByName[f.Name].subFieldByMatch == nil {
			fieldsByName[f.Name].subFieldByMatch = map[string]treeFields{}
		}
		index := fieldsByName[f.Name].subFieldByMatch
		for j := range seq.Content() {
			elem := seq.Content()[j]
			matches := f.Matches[elem]
			str, err := yaml.String(elem, yaml.Trim, yaml.Flow)
			if err != nil {
				return nil, err
			}

			// map the field by the name of the element
			// index the subfields by the matching element so we can put all the fields for the
			// same element under the same branch
			matchKey := strings.Join(matches, "/")
			index[matchKey] = append(index[matchKey], &treeField{name: f.SubName, value: str})
		}
	}

	// iterate over collection of all queried fields in the Resource
	for _, field := range fieldsByName {
		// iterate over collection of elements under the field -- indexed by element name
		for match, subFields := range field.subFieldByMatch {
			// create a new element for this collection of fields
			// note: we will convert name to an index later, but keep the match for sorting
			elem := &treeField{name: match}
			field.matchingElementsAndFields = append(field.matchingElementsAndFields, elem)

			// iterate over collection of queried fields for the element
			for i := range subFields {
				// add to the list of fields for this element
				elem.matchingElementsAndFields = append(elem.matchingElementsAndFields, subFields[i])
			}
		}
		// clear this cached data
		field.subFieldByMatch = nil
	}

	// put the fields in a list so they are ordered
	fieldList := treeFields{}
	for _, v := range fieldsByName {
		fieldList = append(fieldList, v)
	}

	// sort the fields
	sort.Sort(fieldList)
	for i := range fieldList {
		field := fieldList[i]
		// sort the elements under this field
		sort.Sort(field.matchingElementsAndFields)

		for i := range field.matchingElementsAndFields {
			element := field.matchingElementsAndFields[i]
			// sort the elements under a list field by their name
			sort.Sort(element.matchingElementsAndFields)
			// set the name of the element to its index
			element.name = fmt.Sprintf("%d", i)
		}
	}

	return fieldList, nil
}

// treeField wraps a field node
type treeField struct {
	// name is the name of the node
	name string

	// value is the value of the node -- may be empty
	value string

	// matchingElementsAndFields is a slice of fields that go under this as a branch
	matchingElementsAndFields treeFields

	// subFieldByMatch caches matchingElementsAndFields indexed by the name of the matching elem
	subFieldByMatch map[string]treeFields
}

// treeFields wraps a slice of treeField so they can be sorted
type treeFields []*treeField

func (nodes treeFields) Len() int { return len(nodes) }

func (nodes treeFields) Less(i, j int) bool {
	iIndex, iFound := yaml.FieldOrder[nodes[i].name]
	jIndex, jFound := yaml.FieldOrder[nodes[j].name]
	if iFound && jFound {
		return iIndex < jIndex
	}
	if iFound {
		return true
	}
	if jFound {
		return false
	}

	if nodes[i].name != nodes[j].name {
		return nodes[i].name < nodes[j].name
	}
	if nodes[i].value != nodes[j].value {
		return nodes[i].value < nodes[j].value
	}
	return false
}

func (nodes treeFields) Swap(i, j int) { nodes[i], nodes[j] = nodes[j], nodes[i] }
