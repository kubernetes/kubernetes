// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package ext

import (
	"errors"
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"

	"cel.dev/cel-go/cel"
	"cel.dev/cel-go/common/cost"
	"cel.dev/cel-go/common/types"
	"cel.dev/cel-go/common/types/ref"
)

const (
	regexReplace    = "regex.replace"
	regexExtract    = "regex.extract"
	regexExtractAll = "regex.extractAll"
)

// Regex returns a cel.EnvOption to configure extended functions for regular
// expression operations.
//
// Note: all functions use the 'regex' namespace. If you are
// currently using a variable named 'regex', the functions will likely work as
// intended, however there is some chance for collision.
//
// This library depends on the CEL optional type. Please ensure that the
// cel.OptionalTypes() is enabled when using regex extensions.
//
// # Replace
//
// The `regex.replace` function replaces all non-overlapping substring of a regex
// pattern in the target string with a replacement string. Optionally, you can
// limit the number of replacements by providing a count argument. When the count
// is a negative number, the function acts as replace all. Only numeric (\N)
// capture group references are supported in the replacement string, with
// validation for correctness. Backslashed-escaped digits (\1 to \9) within the
// replacement argument can be used to insert text matching the corresponding
// parenthesized group in the regexp pattern. An error will be thrown for invalid
// regex or replace string.
//
//	regex.replace(target: string, pattern: string, replacement: string) -> string
//	regex.replace(target: string, pattern: string, replacement: string, count: int) -> string
//
// Examples:
//
//	regex.replace('hello world hello', 'hello', 'hi') == 'hi world hi'
//	regex.replace('banana', 'a', 'x', 0) == 'banana'
//	regex.replace('banana', 'a', 'x', 1) == 'bxnana'
//	regex.replace('banana', 'a', 'x', 2) == 'bxnxna'
//	regex.replace('banana', 'a', 'x', -12) == 'bxnxnx'
//	regex.replace('foo bar', '(fo)o (ba)r', r'\2 \1') == 'ba fo'
//	regex.replace('test', '(.)', r'\2') \\ Runtime Error invalid replace string
//	regex.replace('foo bar', '(', '$2 $1') \\ Runtime Error invalid regex string
//	regex.replace('id=123', r'id=(?P<value>\d+)', r'value: \values') \\ Runtime Error invalid replace string
//
// # Extract
//
// The `regex.extract` function returns the first match of a regex pattern in a
// string. If no match is found, it returns an optional none value. An error will
// be thrown for invalid regex or for multiple capture groups.
//
//	regex.extract(target: string, pattern: string) -> optional<string>
//
// Examples:
//
//	regex.extract('hello world', 'hello(.*)') == optional.of(' world')
//	regex.extract('item-A, item-B', 'item-(\\w+)') == optional.of('A')
//	regex.extract('HELLO', 'hello') == optional.none()
//	regex.extract('testuser@testdomain', '(.*)@([^.]*)') // Runtime Error multiple capture group
//
// # Extract All
//
// The `regex.extractAll` function returns a list of all matches of a regex
// pattern in a target string. If no matches are found, it returns an empty list. An error will
// be thrown for invalid regex or for multiple capture groups.
//
//	regex.extractAll(target: string, pattern: string) -> list<string>
//
// Examples:
//
//	regex.extractAll('id:123, id:456', 'id:\\d+') == ['id:123', 'id:456']
//	regex.extractAll('id:123, id:456', 'assa') == []
//	regex.extractAll('testuser@testdomain', '(.*)@([^.]*)') // Runtime Error multiple capture group
func Regex(options ...RegexOptions) cel.EnvOption {
	s := &regexLib{
		version: math.MaxUint32,
	}
	for _, o := range options {
		s = o(s)
	}
	return cel.Lib(s)
}

// RegexOptions declares a functional operator for configuring regex extension.
type RegexOptions func(*regexLib) *regexLib

// RegexVersion configures the version of the Regex library definitions to use. See [Regex] for supported values.
func RegexVersion(version uint32) RegexOptions {
	return func(lib *regexLib) *regexLib {
		lib.version = version
		return lib
	}
}

type regexLib struct {
	version uint32
}

// LibraryName implements that SingletonLibrary interface method.
func (r *regexLib) LibraryName() string {
	return "cel.lib.ext.regex"
}

// CompileOptions implements the cel.Library interface method.
func (r *regexLib) CompileOptions() []cel.EnvOption {
	optionalTypesEnabled := func(env *cel.Env) (*cel.Env, error) {
		if !env.HasLibrary("cel.lib.optional") {
			return nil, errors.New("regex library requires the optional library")
		}
		return env, nil
	}
	opts := []cel.EnvOption{
		cel.Function(regexExtract,
			cel.Overload("regex_extract_string_string", []*cel.Type{cel.StringType, cel.StringType}, cel.OptionalType(cel.StringType),
				cel.BinaryBinding(extract))),

		cel.Function(regexExtractAll,
			cel.Overload("regex_extractAll_string_string", []*cel.Type{cel.StringType, cel.StringType}, cel.ListType(cel.StringType),
				cel.BinaryBinding(extractAll))),

		cel.Function(regexReplace,
			cel.Overload("regex_replace_string_string_string", []*cel.Type{cel.StringType, cel.StringType, cel.StringType}, cel.StringType,
				cel.FunctionBinding(regReplace)),
			cel.Overload("regex_replace_string_string_string_int", []*cel.Type{cel.StringType, cel.StringType, cel.StringType, cel.IntType}, cel.StringType,
				cel.FunctionBinding((regReplaceN))),
		),
		cel.CostModel(
			regexExtractCostModel("regex_extract_string_string"),
			regexExtractAllCostModel("regex_extractAll_string_string"),
			regexReplaceCostModel("regex_replace_string_string_string"),
			regexReplaceCostModel("regex_replace_string_string_string_int"),
		),
		cel.EnvOption(optionalTypesEnabled),
	}
	return opts
}

// ProgramOptions implements the cel.Library interface method
func (r *regexLib) ProgramOptions() []cel.ProgramOption {
	return []cel.ProgramOption{}
}

func regReplace(args ...ref.Val) ref.Val {
	target := args[0].(types.String)
	regexStr := args[1].(types.String)
	replaceStr := args[2].(types.String)

	return regReplaceN(target, regexStr, replaceStr, types.Int(-1))
}

func regReplaceN(args ...ref.Val) ref.Val {
	target := string(args[0].(types.String))
	regexStr := string(args[1].(types.String))
	replaceStr := string(args[2].(types.String))
	replaceCount := int64(args[3].(types.Int))

	if replaceCount == 0 {
		return types.String(target)
	}

	// If replaceCount is negative, just do a replaceAll.
	if replaceCount < 0 {
		replaceCount = -1
	}

	re, err := regexp.Compile(regexStr)
	if err != nil {
		return types.WrapErr(err)
	}

	var resultBuilder strings.Builder
	var lastIndex int
	counter := int64(0)

	matches := re.FindAllStringSubmatchIndex(target, -1)

	for _, match := range matches {
		if replaceCount != -1 && counter >= replaceCount {
			break
		}

		processedReplacement, err := replaceStrValidator(target, re, match, replaceStr)
		if err != nil {
			return types.WrapErr(err)
		}

		resultBuilder.WriteString(target[lastIndex:match[0]])
		resultBuilder.WriteString(processedReplacement)
		lastIndex = match[1]
		counter++
	}

	resultBuilder.WriteString(target[lastIndex:])
	return types.String(resultBuilder.String())
}

func replaceStrValidator(target string, re *regexp.Regexp, match []int, replacement string) (string, error) {
	groupCount := re.NumSubexp()
	var sb strings.Builder
	runes := []rune(replacement)

	for i := 0; i < len(runes); i++ {
		c := runes[i]

		if c != '\\' {
			sb.WriteRune(c)
			continue
		}

		if i+1 >= len(runes) {
			return "", fmt.Errorf("invalid replacement string: '%s' \\ not allowed at end", replacement)
		}

		i++
		nextChar := runes[i]

		if nextChar == '\\' {
			sb.WriteRune('\\')
			continue
		}

		groupNum, err := strconv.Atoi(string(nextChar))
		if err != nil {
			return "", fmt.Errorf("invalid replacement string: '%s' \\ must be followed by a digit or \\", replacement)
		}

		if groupNum > groupCount {
			return "", fmt.Errorf("replacement string references group %d but regex has only %d group(s)", groupNum, groupCount)
		}

		if match[2*groupNum] != -1 {
			sb.WriteString(target[match[2*groupNum]:match[2*groupNum+1]])
		}
	}
	return sb.String(), nil
}

func extract(target, regexStr ref.Val) ref.Val {
	t := string(target.(types.String))
	r := string(regexStr.(types.String))
	re, err := regexp.Compile(r)
	if err != nil {
		return types.WrapErr(err)
	}

	if len(re.SubexpNames())-1 > 1 {
		return types.WrapErr(fmt.Errorf("regular expression has more than one capturing group: %q", r))
	}

	matches := re.FindStringSubmatch(t)
	if len(matches) == 0 {
		return types.OptionalNone
	}

	// If there is a capturing group, return the first match; otherwise, return the whole match.
	if len(matches) > 1 {
		capturedGroup := matches[1]
		// If optional group is empty, return OptionalNone.
		if capturedGroup == "" {
			return types.OptionalNone
		}
		return types.OptionalOf(types.String(capturedGroup))
	}
	return types.OptionalOf(types.String(matches[0]))
}

func extractAll(target, regexStr ref.Val) ref.Val {
	t := string(target.(types.String))
	r := string(regexStr.(types.String))
	re, err := regexp.Compile(r)
	if err != nil {
		return types.WrapErr(err)
	}

	groupCount := len(re.SubexpNames()) - 1
	if groupCount > 1 {
		return types.WrapErr(fmt.Errorf("regular expression has more than one capturing group: %q", r))
	}

	matches := re.FindAllStringSubmatch(t, -1)
	result := make([]string, 0, len(matches))
	if len(matches) == 0 {
		return types.NewStringList(types.DefaultTypeAdapter, result)
	}

	if groupCount != 1 {
		for _, match := range matches {
			result = append(result, match[0])
		}
		return types.NewStringList(types.DefaultTypeAdapter, result)
	}

	for _, match := range matches {
		if match[1] != "" {
			result = append(result, match[1])
		}
	}
	return types.NewStringList(types.DefaultTypeAdapter, result)
}

func estimateRegexSearchCost(c cost.Estimator, args []cost.AstNode) (cost.SizeEstimate, cost.CostEstimate) {
	targetSize := cost.EstimateSize(c, args[0])
	// Fixed size estimate of +1 is added for safety from zero size args.
	// The target cost is the size of the target string, scaled by a traversal factor.
	targetCost := targetSize.Add(cost.FixedSizeEstimate(1)).MultiplyByCostFactor(cost.StringTraversalCostFactor)
	// The regex cost is the size of the regex pattern, scaled by a complexity factor.
	regexCost := cost.EstimateSize(c, args[1]).Add(cost.FixedSizeEstimate(1)).MultiplyByCostFactor(cost.RegexStringLengthCostFactor)
	return targetSize, targetCost.Multiply(regexCost)
}

func estimateExtractCost(c cost.Estimator, target *cost.AstNode, args []cost.AstNode) *cost.CallEstimate {
	if len(args) != 2 {
		return nil
	}
	targetSize, searchCost := estimateRegexSearchCost(c, args)
	// The result is a single string. Worst Case: it's the size of the entire target.
	resultSize := cost.RangedSizeEstimate(0, targetSize.Max)
	// The total cost is the search cost plus the allocation cost for the result string.
	return cost.NewCallEstimate(searchCost.Add(resultSize.AsCost()), &resultSize)
}

func estimateExtractAllCost(c cost.Estimator, target *cost.AstNode, args []cost.AstNode) *cost.CallEstimate {
	if len(args) != 2 {
		return nil
	}
	targetSize, searchCost := estimateRegexSearchCost(c, args)
	// The result is a list of strings. Worst Case: its contents are the size of the entire target.
	elemSize := cost.RangedSizeEstimate(0, targetSize.Max)
	resultSize := cost.ListSizeEstimate(cost.RangedSizeEstimate(0, targetSize.Max), elemSize)
	// The cost to allocate the result list is its base cost plus the size of its contents.
	allocationSize := resultSize.Add(cost.FixedSizeEstimate(cost.ListCreateBaseCost))
	// The total cost is the search cost plus the allocation cost for the result list.
	return cost.NewCallEstimate(searchCost.Add(allocationSize.AsCost()), &resultSize)
}

func estimateReplaceCost(c cost.Estimator, target *cost.AstNode, args []cost.AstNode) *cost.CallEstimate {
	if target != nil || (len(args) != 3 && len(args) != 4) {
		return nil
	}
	targetSize, searchCost := estimateRegexSearchCost(c, args)
	replacementSize := cost.EstimateSize(c, args[2])
	// Estimate the potential size range of the output string. The final size could be smaller
	// (if the replacement size is 0) or larger than the original.
	allReplacedSize := cost.SafeMultiply(targetSize.Max, replacementSize.Max)
	noneReplacedSize := targetSize.Max
	resultSize := cost.RangedSizeEstimate(noneReplacedSize, allReplacedSize)
	if replacementSize.Max == 0 {
		resultSize = cost.RangedSizeEstimate(allReplacedSize, noneReplacedSize)
	}
	// The final cost is the search cost plus the allocation cost for the output string.
	return cost.NewCallEstimate(searchCost.Add(resultSize.AsCost()), &resultSize)
}

func trackReplaceCost(args []ref.Val, result ref.Val) *uint64 {
	targetCost := float64(cost.SafeAdd(cost.ActualSize(args[0]), 1)) * cost.StringTraversalCostFactor
	regexCost := float64(cost.SafeAdd(cost.ActualSize(args[1]), 1)) * cost.RegexStringLengthCostFactor
	totalCost := float64(cost.CallCost) + targetCost*regexCost + float64(cost.ActualSize(result))
	// Convert to uint64 for the final cost. Unlike the extract trackers this truncates; the
	// two are reconciled in cost model version 1 to avoid raising the ModelVersion0 charge.
	finalCost := cost.SafeTrunc(totalCost)
	return &finalCost
}

// regexSearch is the cost of matching pattern against target: the product of their lengths.
func regexSearch(target, pattern cost.QuantityExpr) cost.QuantityExpr {
	return cost.Mul(
		cost.Scale(cost.Sum(cost.Length(target), cost.Const(1)), cost.StringTraversalCostFactor),
		cost.Scale(cost.Sum(cost.Length(pattern), cost.Const(1)), cost.RegexStringLengthCostFactor),
	)
}

// regexExtractCostModel models regex.extract: it searches the target, and returns an optional
// holding the first match or capture group, a part of the target. The result is charged as the
// optional it is, 1 plus the size of the part (1 when nothing matched); see cost.Optional. Version 0
// sizes an optional as 1, which is what v0.32.0 charged.
func regexExtractCostModel(id string) cost.OverloadModel {
	return cost.Overload(id,
		cost.EvalCost(cost.Sum(cost.Const(cost.CallCost), regexSearch(cost.Arg(0), cost.Arg(1)), cost.Result())),
		cost.ResultSize(cost.Optional(cost.AtMost(cost.Arg(0)))),
		cost.BeforeVersion(1, cost.ManualEstimator(estimateExtractCost)),
	)
}

// regexExtractAllCostModel models regex.extractAll: it returns parts of the target, sized as the
// parts of a split.
func regexExtractAllCostModel(id string) cost.OverloadModel {
	return cost.Overload(id,
		cost.EvalCost(cost.AllocList(regexSearch(cost.Arg(0), cost.Arg(1)), cost.Length(cost.Result()))),
		cost.ResultSize(splitSize(cost.Arg(0))),
		cost.BeforeVersion(1, cost.ManualEstimator(estimateExtractAllCost)),
	)
}

// regexReplaceCostModel models regex.replace: it may expand each of the at most n + 1 matches into
// the replacement, whose capture group references may each expand to the whole target.
func regexReplaceCostModel(id string) cost.OverloadModel {
	target := cost.Arg(0)
	return cost.Overload(id,
		cost.EvalCost(cost.Sum(cost.Const(cost.CallCost), regexSearch(target, cost.Arg(1)), cost.Length(cost.Result()))),
		cost.ResultSize(cost.AtMost(cost.Sum(target,
			cost.Mul(cost.Sum(target, cost.Const(1)), cost.Arg(2), cost.Sum(target, cost.Const(1)))))),
		cost.BeforeVersion(1, cost.ManualCost(estimateReplaceCost, trackReplaceCost)),
	)
}

