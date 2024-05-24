/*
Copyright 2023 The Kubernetes Authors.

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

package knftables

import (
	"fmt"
	"io"
	"strings"
)

// Object implementation for Table
func (table *Table) validate(verb verb) error {
	switch verb {
	case addVerb, createVerb, flushVerb:
		if table.Handle != nil {
			return fmt.Errorf("cannot specify Handle in %s operation", verb)
		}
	case deleteVerb:
		// Handle can be nil or non-nil
	default:
		return fmt.Errorf("%s is not implemented for tables", verb)
	}

	return nil
}

func (table *Table) writeOperation(verb verb, ctx *nftContext, writer io.Writer) {
	// Special case for delete-by-handle
	if verb == deleteVerb && table.Handle != nil {
		fmt.Fprintf(writer, "delete table %s handle %d", ctx.family, *table.Handle)
		return
	}

	// All other cases refer to the table by name
	fmt.Fprintf(writer, "%s table %s %s", verb, ctx.family, ctx.table)
	if verb == addVerb || verb == createVerb {
		if table.Comment != nil && !ctx.noObjectComments {
			fmt.Fprintf(writer, " { comment %q ; }", *table.Comment)
		}
	}
	fmt.Fprintf(writer, "\n")
}

// Object implementation for Chain
func (chain *Chain) validate(verb verb) error {
	if chain.Hook == nil {
		if chain.Type != nil || chain.Priority != nil {
			return fmt.Errorf("regular chain %q must not specify Type or Priority", chain.Name)
		}
		if chain.Device != nil {
			return fmt.Errorf("regular chain %q must not specify Device", chain.Name)
		}
	} else {
		if chain.Type == nil || chain.Priority == nil {
			return fmt.Errorf("base chain %q must specify Type and Priority", chain.Name)
		}
	}

	switch verb {
	case addVerb, createVerb, flushVerb:
		if chain.Name == "" {
			return fmt.Errorf("no name specified for chain")
		}
		if chain.Handle != nil {
			return fmt.Errorf("cannot specify Handle in %s operation", verb)
		}
	case deleteVerb:
		if chain.Name == "" && chain.Handle == nil {
			return fmt.Errorf("must specify either name or handle")
		}
	default:
		return fmt.Errorf("%s is not implemented for chains", verb)
	}

	return nil
}

func (chain *Chain) writeOperation(verb verb, ctx *nftContext, writer io.Writer) {
	// Special case for delete-by-handle
	if verb == deleteVerb && chain.Handle != nil {
		fmt.Fprintf(writer, "delete chain %s %s handle %d", ctx.family, ctx.table, *chain.Handle)
		return
	}

	fmt.Fprintf(writer, "%s chain %s %s %s", verb, ctx.family, ctx.table, chain.Name)
	if verb == addVerb || verb == createVerb {
		if chain.Type != nil || (chain.Comment != nil && !ctx.noObjectComments) {
			fmt.Fprintf(writer, " {")

			if chain.Type != nil {
				fmt.Fprintf(writer, " type %s hook %s", *chain.Type, *chain.Hook)
				if chain.Device != nil {
					fmt.Fprintf(writer, " device %q", *chain.Device)
				}

				// Parse the priority to a number if we can, because older
				// versions of nft don't accept certain named priorities
				// in all contexts (eg, "dstnat" priority in the "output"
				// hook).
				if priority, err := ParsePriority(ctx.family, string(*chain.Priority)); err == nil {
					fmt.Fprintf(writer, " priority %d ;", priority)
				} else {
					fmt.Fprintf(writer, " priority %s ;", *chain.Priority)
				}
			}
			if chain.Comment != nil && !ctx.noObjectComments {
				fmt.Fprintf(writer, " comment %q ;", *chain.Comment)
			}

			fmt.Fprintf(writer, " }")
		}
	}

	fmt.Fprintf(writer, "\n")
}

// Object implementation for Rule
func (rule *Rule) validate(verb verb) error {
	if rule.Chain == "" {
		return fmt.Errorf("no chain name specified for rule")
	}

	if rule.Index != nil && rule.Handle != nil {
		return fmt.Errorf("cannot specify both Index and Handle")
	}

	switch verb {
	case addVerb, insertVerb:
		if rule.Rule == "" {
			return fmt.Errorf("no rule specified")
		}
	case replaceVerb:
		if rule.Rule == "" {
			return fmt.Errorf("no rule specified")
		}
		if rule.Handle == nil {
			return fmt.Errorf("must specify Handle with %s", verb)
		}
	case deleteVerb:
		if rule.Handle == nil {
			return fmt.Errorf("must specify Handle with %s", verb)
		}
	default:
		return fmt.Errorf("%s is not implemented for rules", verb)
	}

	return nil
}

func (rule *Rule) writeOperation(verb verb, ctx *nftContext, writer io.Writer) {
	fmt.Fprintf(writer, "%s rule %s %s %s", verb, ctx.family, ctx.table, rule.Chain)
	if rule.Index != nil {
		fmt.Fprintf(writer, " index %d", *rule.Index)
	} else if rule.Handle != nil {
		fmt.Fprintf(writer, " handle %d", *rule.Handle)
	}

	switch verb {
	case addVerb, insertVerb, replaceVerb:
		fmt.Fprintf(writer, " %s", rule.Rule)

		if rule.Comment != nil {
			fmt.Fprintf(writer, " comment %q", *rule.Comment)
		}
	}

	fmt.Fprintf(writer, "\n")
}

// Object implementation for Set
func (set *Set) validate(verb verb) error {
	switch verb {
	case addVerb, createVerb:
		if (set.Type == "" && set.TypeOf == "") || (set.Type != "" && set.TypeOf != "") {
			return fmt.Errorf("set must specify either Type or TypeOf")
		}
		if set.Handle != nil {
			return fmt.Errorf("cannot specify Handle in %s operation", verb)
		}
		fallthrough
	case flushVerb:
		if set.Name == "" {
			return fmt.Errorf("no name specified for set")
		}
	case deleteVerb:
		if set.Name == "" && set.Handle == nil {
			return fmt.Errorf("must specify either name or handle")
		}
	default:
		return fmt.Errorf("%s is not implemented for sets", verb)
	}

	return nil
}

func (set *Set) writeOperation(verb verb, ctx *nftContext, writer io.Writer) {
	// Special case for delete-by-handle
	if verb == deleteVerb && set.Handle != nil {
		fmt.Fprintf(writer, "delete set %s %s handle %d", ctx.family, ctx.table, *set.Handle)
		return
	}

	fmt.Fprintf(writer, "%s set %s %s %s", verb, ctx.family, ctx.table, set.Name)
	if verb == addVerb || verb == createVerb {
		fmt.Fprintf(writer, " {")

		if set.Type != "" {
			fmt.Fprintf(writer, " type %s ;", set.Type)
		} else {
			fmt.Fprintf(writer, " typeof %s ;", set.TypeOf)
		}

		if len(set.Flags) != 0 {
			fmt.Fprintf(writer, " flags ")
			for i := range set.Flags {
				if i > 0 {
					fmt.Fprintf(writer, ",")
				}
				fmt.Fprintf(writer, "%s", set.Flags[i])
			}
			fmt.Fprintf(writer, " ;")
		}

		if set.Timeout != nil {
			fmt.Fprintf(writer, " timeout %ds ;", int64(set.Timeout.Seconds()))
		}
		if set.GCInterval != nil {
			fmt.Fprintf(writer, " gc-interval %ds ;", int64(set.GCInterval.Seconds()))
		}
		if set.Size != nil {
			fmt.Fprintf(writer, " size %d ;", *set.Size)
		}
		if set.Policy != nil {
			fmt.Fprintf(writer, " policy %s ;", *set.Policy)
		}
		if set.AutoMerge != nil && *set.AutoMerge {
			fmt.Fprintf(writer, " auto-merge ;")
		}

		if set.Comment != nil && !ctx.noObjectComments {
			fmt.Fprintf(writer, " comment %q ;", *set.Comment)
		}

		fmt.Fprintf(writer, " }")
	}

	fmt.Fprintf(writer, "\n")
}

// Object implementation for Map
func (mapObj *Map) validate(verb verb) error {
	switch verb {
	case addVerb, createVerb:
		if (mapObj.Type == "" && mapObj.TypeOf == "") || (mapObj.Type != "" && mapObj.TypeOf != "") {
			return fmt.Errorf("map must specify either Type or TypeOf")
		}
		if mapObj.Handle != nil {
			return fmt.Errorf("cannot specify Handle in %s operation", verb)
		}
		fallthrough
	case flushVerb:
		if mapObj.Name == "" {
			return fmt.Errorf("no name specified for map")
		}
	case deleteVerb:
		if mapObj.Name == "" && mapObj.Handle == nil {
			return fmt.Errorf("must specify either name or handle")
		}
	default:
		return fmt.Errorf("%s is not implemented for maps", verb)
	}

	return nil
}

func (mapObj *Map) writeOperation(verb verb, ctx *nftContext, writer io.Writer) {
	// Special case for delete-by-handle
	if verb == deleteVerb && mapObj.Handle != nil {
		fmt.Fprintf(writer, "delete map %s %s handle %d", ctx.family, ctx.table, *mapObj.Handle)
		return
	}

	fmt.Fprintf(writer, "%s map %s %s %s", verb, ctx.family, ctx.table, mapObj.Name)
	if verb == addVerb || verb == createVerb {
		fmt.Fprintf(writer, " {")

		if mapObj.Type != "" {
			fmt.Fprintf(writer, " type %s ;", mapObj.Type)
		} else {
			fmt.Fprintf(writer, " typeof %s ;", mapObj.TypeOf)
		}

		if len(mapObj.Flags) != 0 {
			fmt.Fprintf(writer, " flags ")
			for i := range mapObj.Flags {
				if i > 0 {
					fmt.Fprintf(writer, ",")
				}
				fmt.Fprintf(writer, "%s", mapObj.Flags[i])
			}
			fmt.Fprintf(writer, " ;")
		}

		if mapObj.Timeout != nil {
			fmt.Fprintf(writer, " timeout %ds ;", int64(mapObj.Timeout.Seconds()))
		}
		if mapObj.GCInterval != nil {
			fmt.Fprintf(writer, " gc-interval %ds ;", int64(mapObj.GCInterval.Seconds()))
		}
		if mapObj.Size != nil {
			fmt.Fprintf(writer, " size %d ;", *mapObj.Size)
		}
		if mapObj.Policy != nil {
			fmt.Fprintf(writer, " policy %s ;", *mapObj.Policy)
		}

		if mapObj.Comment != nil && !ctx.noObjectComments {
			fmt.Fprintf(writer, " comment %q ;", *mapObj.Comment)
		}

		fmt.Fprintf(writer, " }")
	}

	fmt.Fprintf(writer, "\n")
}

// Object implementation for Element
func (element *Element) validate(verb verb) error {
	if element.Map == "" && element.Set == "" {
		return fmt.Errorf("no set/map name specified for element")
	} else if element.Set != "" && element.Map != "" {
		return fmt.Errorf("element specifies both a set name and a map name")
	}

	if len(element.Key) == 0 {
		return fmt.Errorf("no key specified for element")
	}
	if element.Set != "" && len(element.Value) != 0 {
		return fmt.Errorf("map value specified for set element")
	}

	switch verb {
	case addVerb, createVerb:
		if element.Map != "" && len(element.Value) == 0 {
			return fmt.Errorf("no map value specified for map element")
		}
	case deleteVerb:
	default:
		return fmt.Errorf("%s is not implemented for elements", verb)
	}

	return nil
}

func (element *Element) writeOperation(verb verb, ctx *nftContext, writer io.Writer) {
	name := element.Set
	if name == "" {
		name = element.Map
	}

	fmt.Fprintf(writer, "%s element %s %s %s { %s", verb, ctx.family, ctx.table, name,
		strings.Join(element.Key, " . "))

	if verb == addVerb || verb == createVerb {
		if element.Comment != nil {
			fmt.Fprintf(writer, " comment %q", *element.Comment)
		}

		if len(element.Value) != 0 {
			fmt.Fprintf(writer, " : %s", strings.Join(element.Value, " . "))
		}
	}

	fmt.Fprintf(writer, " }\n")
}
