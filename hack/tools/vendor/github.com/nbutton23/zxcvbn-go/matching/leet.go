package matching

import (
	"strings"

	"github.com/nbutton23/zxcvbn-go/entropy"
	"github.com/nbutton23/zxcvbn-go/match"
)

// L33TMatcherName id
const L33TMatcherName = "l33t"

//FilterL33tMatcher can be pass to zxcvbn-go.PasswordStrength to skip that matcher
func FilterL33tMatcher(m match.Matcher) bool {
	return m.ID == L33TMatcherName
}

func l33tMatch(password string) []match.Match {
	permutations := getPermutations(password)

	var matches []match.Match

	for _, permutation := range permutations {
		for _, mather := range dictionaryMatchers {
			matches = append(matches, mather.MatchingFunc(permutation)...)
		}
	}

	for _, match := range matches {
		match.Entropy += entropy.ExtraLeetEntropy(match, password)
		match.DictionaryName = match.DictionaryName + "_3117"
	}

	return matches
}

// This function creates a list of permutations based on a fixed table stored on data. The table
// will be reduced in order to proceed in the function using only relevant values (see
// relevantL33tSubtable).
func getPermutations(password string) []string {
	substitutions := relevantL33tSubtable(password)
	permutations := getAllPermutationsOfLeetSubstitutions(password, substitutions)
	return permutations
}

// This function loads the table from data but only keep in memory the values that are present
// inside the provided password.
func relevantL33tSubtable(password string) map[string][]string {
	relevantSubs := make(map[string][]string)
	for key, values := range l33tTable.Graph {
		for _, value := range values {
			if strings.Contains(password, value) {
				relevantSubs[key] = append(relevantSubs[key], value)
			}
		}
	}

	return relevantSubs
}

// This function creates the list of permutations of a given password using the provided table as
// reference for its operation.
func getAllPermutationsOfLeetSubstitutions(password string, table map[string][]string) []string {
	result := []string{}

	// create a list of tables without conflicting keys/values (this happens for "|", "7" and "1")
	noConflictsTables := createListOfMapsWithoutConflicts(table)
	for _, noConflictsTable := range noConflictsTables {
		substitutionsMaps := createSubstitutionsMapsFromTable(noConflictsTable)
		for _, substitutionsMap := range substitutionsMaps {
			newValue := createWordForSubstitutionMap(password, substitutionsMap)
			if !stringSliceContainsValue(result, newValue) {
				result = append(result, newValue)
			}
		}
	}

	return result
}

// Create the possible list of maps removing the conflicts from it. As an example, the value "|"
// may represent "i" and "l". For each representation of the conflicting value, a new map is
// created. This may grow exponencialy according to the number of conflicts. The number of maps
// returned by this function may be reduced if the relevantL33tSubtable function was called to
// identify only relevant items.
func createListOfMapsWithoutConflicts(table map[string][]string) []map[string][]string {
	// the resulting list starts with the provided table
	result := []map[string][]string{}
	result = append(result, table)

	// iterate over the list of conflicts in order to expand the maps for each one
	conflicts := retrieveConflictsListFromTable(table)
	for _, value := range conflicts {
		newMapList := []map[string][]string{}

		// for each conflict a new list of maps will be created for every already known map
		for _, currentMap := range result {
			newMaps := createDifferentMapsForLeetChar(currentMap, value)
			newMapList = append(newMapList, newMaps...)
		}

		result = newMapList
	}

	return result
}

// This function retrieves the list of values that appear for one or more keys. This is usefull to
// know which l33t chars can represent more than one letter.
func retrieveConflictsListFromTable(table map[string][]string) []string {
	result := []string{}
	foundValues := []string{}

	for _, values := range table {
		for _, value := range values {
			if stringSliceContainsValue(foundValues, value) {
				// only add on results if it was not identified as conflict before
				if !stringSliceContainsValue(result, value) {
					result = append(result, value)
				}
			} else {
				foundValues = append(foundValues, value)
			}
		}
	}

	return result
}

// This function aims to create different maps for a given char if this char represents a conflict.
// If the specified char is not a conflit one, the same map will be returned. In scenarios which
// the provided char can not be found on map, an empty list will be returned. This function was
// designed to be used on conflicts situations.
func createDifferentMapsForLeetChar(table map[string][]string, leetChar string) []map[string][]string {
	result := []map[string][]string{}

	keysWithSameValue := retrieveListOfKeysWithSpecificValueFromTable(table, leetChar)
	for _, key := range keysWithSameValue {
		newMap := copyMapRemovingSameValueFromOtherKeys(table, key, leetChar)
		result = append(result, newMap)
	}

	return result
}

// This function retrieves the list of keys that can be represented using the given value.
func retrieveListOfKeysWithSpecificValueFromTable(table map[string][]string, valueToFind string) []string {
	result := []string{}

	for key, values := range table {
		for _, value := range values {
			if value == valueToFind && !stringSliceContainsValue(result, key) {
				result = append(result, key)
			}
		}
	}

	return result
}

// This function returns a lsit of substitution map from a given table. Each map in the result will
// provide only one representation for each value. As an example, if the provided map contains the
// values "@" and "4" in the possibilities to represent "a", two maps will be created where one
// will contain "a" mapping to "@" and the other one will provide "a" mapping to "4".
func createSubstitutionsMapsFromTable(table map[string][]string) []map[string]string {
	result := []map[string]string{{"": ""}}

	for key, values := range table {
		newResult := []map[string]string{}

		for _, mapInCurrentResult := range result {
			for _, value := range values {
				newMapForValue := copyMap(mapInCurrentResult)
				newMapForValue[key] = value
				newResult = append(newResult, newMapForValue)
			}
		}

		result = newResult
	}

	// verification to make sure that the slice was filled
	if len(result) == 1 && len(result[0]) == 1 && result[0][""] == "" {
		return []map[string]string{}
	}

	return result
}

// This function replaces the values provided on substitution map over the provided word.
func createWordForSubstitutionMap(word string, substitutionMap map[string]string) string {
	result := word
	for key, value := range substitutionMap {
		result = strings.Replace(result, value, key, -1)
	}

	return result
}

func stringSliceContainsValue(slice []string, value string) bool {
	for _, valueInSlice := range slice {
		if valueInSlice == value {
			return true
		}
	}

	return false
}

func copyMap(table map[string]string) map[string]string {
	result := make(map[string]string)

	for key, value := range table {
		result[key] = value
	}

	return result
}

// This function creates a new map based on the one provided but excluding possible representations
// of the same value on other keys.
func copyMapRemovingSameValueFromOtherKeys(table map[string][]string, keyToFix string, valueToFix string) map[string][]string {
	result := make(map[string][]string)

	for key, values := range table {
		for _, value := range values {
			if !(value == valueToFix && key != keyToFix) {
				result[key] = append(result[key], value)
			}
		}
	}

	return result
}
