// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
package query

import (
	"fmt"
	"log"
	"strings"
)

// https://docs.microsoft.com/en-us/windows/win32/wmisdk/wql-sql-for-wmi

type CompareOperator string

const (
	Equals            CompareOperator = "="
	LessThan          CompareOperator = "<"
	GreaterThan       CompareOperator = ">"
	LessThanEquals    CompareOperator = "<="
	GreaterThanEquals CompareOperator = ">="
	NotEquals         CompareOperator = "<>"
	Like              CompareOperator = "LIKE"
	Isa               CompareOperator = "ISA"
)

type WmiQueryFilter struct {
	Name     string
	Value    string
	Operator CompareOperator
}

type WmiQuery struct {
	ClassName  string
	Filters    []*WmiQueryFilter
	SelectList []string
}

func NewWmiQuery(className string, filters ...string) (wquery *WmiQuery) {
	wquery = &WmiQuery{ClassName: className, Filters: []*WmiQueryFilter{}}
	if len(filters) == 0 {
		return
	}

	wquery.BuildQueryFilter(filters)
	return
}

func NewWmiQueryWithSelectList(className string, selectList []string, filters ...string) (wquery *WmiQuery) {
	wquery = &WmiQuery{ClassName: className, SelectList: selectList, Filters: []*WmiQueryFilter{}}
	if len(filters) == 0 {
		return
	}

	wquery.BuildQueryFilter(filters)
	return
}

func (q *WmiQuery) BuildQueryFilter(filters []string) {
	if len(filters)%2 == 1 {
		log.Fatalf("Even number of strings is required to build key=value set of filters: [%+v]\n", filters)
	}

	for i := 0; i < len(filters); i = i + 2 {
		qfilter := NewWmiQueryFilter(filters[i], filters[i+1], Equals)
		q.Filters = append(q.Filters, qfilter)
	}

	return
}

// NewWmiQueryFilter
func NewWmiQueryFilter(name, value string, oper CompareOperator) *WmiQueryFilter {
	return &WmiQueryFilter{Name: name, Value: value, Operator: oper}
}

func (q *WmiQueryFilter) String() string {
	if q.Operator == Like {
		return fmt.Sprintf("%s %s '%%%s%%'", q.Name, q.Operator, q.Value)
	} else {
		return fmt.Sprintf("%s %s '%s'", q.Name, q.Operator, q.Value)
	}
}
func (q *WmiQuery) AddFilterWithComparer(propertyName, value string, oper CompareOperator) {
	q.Filters = append(q.Filters, NewWmiQueryFilter(propertyName, value, oper))
	return
}
func (q *WmiQuery) AddFilter(propertyName, value string) {
	q.Filters = append(q.Filters, NewWmiQueryFilter(propertyName, value, Equals))
	return
}

// HasFilter
func (q *WmiQuery) HasFilter() bool {
	return len(q.Filters) > 0
}

// String
func (q *WmiQuery) String() (queryString string) {
	paramStr := "*"
	if len(q.SelectList) > 0 {
		paramStr = strings.Join(q.SelectList, ",")
	}
	queryString = fmt.Sprintf("SELECT %s FROM %s", paramStr, q.ClassName)

	if len(q.Filters) == 0 {
		return
	}

	queryString = fmt.Sprintf("%s WHERE ", queryString)

	for _, val := range q.Filters[:len(q.Filters)-1] {
		queryString = queryString + fmt.Sprintf(" %s AND", val.String())
	}

	queryString = queryString + fmt.Sprintf(" %s ", q.Filters[len(q.Filters)-1].String())
	return
}

type WmiQueryFilterCollection []*WmiQueryFilter

func (c *WmiQueryFilterCollection) String() string {
	queryString := ""
	for _, query := range *c {
		queryString = fmt.Sprintf("%s AND %s", queryString, query.String())
	}
	return queryString
}
