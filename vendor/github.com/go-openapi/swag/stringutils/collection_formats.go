// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package stringutils

import "strings"

const (
	// collectionFormatComma = "csv"
	collectionFormatSpace = "ssv"
	collectionFormatTab   = "tsv"
	collectionFormatPipe  = "pipes"
	collectionFormatMulti = "multi"

	collectionFormatDefaultSep = ","
)

// JoinByFormat joins a string array by a known format (e.g. swagger's collectionFormat attribute):
//
//	ssv: space separated value
//	tsv: tab separated value
//	pipes: pipe (|) separated value
//	csv: comma separated value (default)
func JoinByFormat(data []string, format string) []string {
	if len(data) == 0 {
		return data
	}
	var sep string
	switch format {
	case collectionFormatSpace:
		sep = " "
	case collectionFormatTab:
		sep = "\t"
	case collectionFormatPipe:
		sep = "|"
	case collectionFormatMulti:
		return data
	default:
		sep = collectionFormatDefaultSep
	}
	return []string{strings.Join(data, sep)}
}

// SplitByFormat splits a string by a known format:
//
//	ssv: space separated value
//	tsv: tab separated value
//	pipes: pipe (|) separated value
//	csv: comma separated value (default)
func SplitByFormat(data, format string) []string {
	if data == "" {
		return nil
	}
	var sep string
	switch format {
	case collectionFormatSpace:
		sep = " "
	case collectionFormatTab:
		sep = "\t"
	case collectionFormatPipe:
		sep = "|"
	case collectionFormatMulti:
		return nil
	default:
		sep = collectionFormatDefaultSep
	}
	var result []string
	for _, s := range strings.Split(data, sep) {
		if ts := strings.TrimSpace(s); ts != "" {
			result = append(result, ts)
		}
	}
	return result
}
