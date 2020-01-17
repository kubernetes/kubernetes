package objx

import (
	"regexp"
	"strconv"
	"strings"
)

const (
	// PathSeparator is the character used to separate the elements
	// of the keypath.
	//
	// For example, `location.address.city`
	PathSeparator string = "."

	// arrayAccesRegexString is the regex used to extract the array number
	// from the access path
	arrayAccesRegexString = `^(.+)\[([0-9]+)\]$`
)

// arrayAccesRegex is the compiled arrayAccesRegexString
var arrayAccesRegex = regexp.MustCompile(arrayAccesRegexString)

// Get gets the value using the specified selector and
// returns it inside a new Obj object.
//
// If it cannot find the value, Get will return a nil
// value inside an instance of Obj.
//
// Get can only operate directly on map[string]interface{} and []interface.
//
// Example
//
// To access the title of the third chapter of the second book, do:
//
//    o.Get("books[1].chapters[2].title")
func (m Map) Get(selector string) *Value {
	rawObj := access(m, selector, nil, false)
	return &Value{data: rawObj}
}

// Set sets the value using the specified selector and
// returns the object on which Set was called.
//
// Set can only operate directly on map[string]interface{} and []interface
//
// Example
//
// To set the title of the third chapter of the second book, do:
//
//    o.Set("books[1].chapters[2].title","Time to Go")
func (m Map) Set(selector string, value interface{}) Map {
	access(m, selector, value, true)
	return m
}

// getIndex returns the index, which is hold in s by two braches.
// It also returns s withour the index part, e.g. name[1] will return (1, name).
// If no index is found, -1 is returned
func getIndex(s string) (int, string) {
	arrayMatches := arrayAccesRegex.FindStringSubmatch(s)
	if len(arrayMatches) > 0 {
		// Get the key into the map
		selector := arrayMatches[1]
		// Get the index into the array at the key
		// We know this cannt fail because arrayMatches[2] is an int for sure
		index, _ := strconv.Atoi(arrayMatches[2])
		return index, selector
	}
	return -1, s
}

// access accesses the object using the selector and performs the
// appropriate action.
func access(current interface{}, selector string, value interface{}, isSet bool) interface{} {
	selSegs := strings.SplitN(selector, PathSeparator, 2)
	thisSel := selSegs[0]
	index := -1

	if strings.Contains(thisSel, "[") {
		index, thisSel = getIndex(thisSel)
	}

	if curMap, ok := current.(Map); ok {
		current = map[string]interface{}(curMap)
	}
	// get the object in question
	switch current.(type) {
	case map[string]interface{}:
		curMSI := current.(map[string]interface{})
		if len(selSegs) <= 1 && isSet {
			curMSI[thisSel] = value
			return nil
		}

		_, ok := curMSI[thisSel].(map[string]interface{})
		if (curMSI[thisSel] == nil || !ok) && index == -1 && isSet {
			curMSI[thisSel] = map[string]interface{}{}
		}

		current = curMSI[thisSel]
	default:
		current = nil
	}
	// do we need to access the item of an array?
	if index > -1 {
		if array, ok := current.([]interface{}); ok {
			if index < len(array) {
				current = array[index]
			} else {
				current = nil
			}
		}
	}
	if len(selSegs) > 1 {
		current = access(current, selSegs[1], value, isSet)
	}
	return current
}
