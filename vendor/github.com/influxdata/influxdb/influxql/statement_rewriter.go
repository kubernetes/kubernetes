package influxql

import "errors"

// RewriteStatement rewrites stmt into a new statement, if applicable.
func RewriteStatement(stmt Statement) (Statement, error) {
	switch stmt := stmt.(type) {
	case *ShowFieldKeysStatement:
		return rewriteShowFieldKeysStatement(stmt)
	case *ShowMeasurementsStatement:
		return rewriteShowMeasurementsStatement(stmt)
	case *ShowSeriesStatement:
		return rewriteShowSeriesStatement(stmt)
	case *ShowTagKeysStatement:
		return rewriteShowTagKeysStatement(stmt)
	case *ShowTagValuesStatement:
		return rewriteShowTagValuesStatement(stmt)
	default:
		return stmt, nil
	}
}

func rewriteShowFieldKeysStatement(stmt *ShowFieldKeysStatement) (Statement, error) {
	return &SelectStatement{
		Fields: Fields([]*Field{
			{Expr: &VarRef{Val: "fieldKey"}},
			{Expr: &VarRef{Val: "fieldType"}},
		}),
		Sources:    rewriteSources(stmt.Sources, "_fieldKeys", stmt.Database),
		Condition:  rewriteSourcesCondition(stmt.Sources, nil),
		Offset:     stmt.Offset,
		Limit:      stmt.Limit,
		SortFields: stmt.SortFields,
		OmitTime:   true,
		Dedupe:     true,
	}, nil
}

func rewriteShowMeasurementsStatement(stmt *ShowMeasurementsStatement) (Statement, error) {
	// Check for time in WHERE clause (not supported).
	if HasTimeExpr(stmt.Condition) {
		return nil, errors.New("SHOW MEASUREMENTS doesn't support time in WHERE clause")
	}

	condition := stmt.Condition
	if stmt.Source != nil {
		condition = rewriteSourcesCondition(Sources([]Source{stmt.Source}), stmt.Condition)
	}
	return &ShowMeasurementsStatement{
		Database:   stmt.Database,
		Condition:  condition,
		Limit:      stmt.Limit,
		Offset:     stmt.Offset,
		SortFields: stmt.SortFields,
	}, nil
}

func rewriteShowSeriesStatement(stmt *ShowSeriesStatement) (Statement, error) {
	// Check for time in WHERE clause (not supported).
	if HasTimeExpr(stmt.Condition) {
		return nil, errors.New("SHOW SERIES doesn't support time in WHERE clause")
	}

	return &SelectStatement{
		Fields: []*Field{
			{Expr: &VarRef{Val: "key"}},
		},
		Sources:    rewriteSources(stmt.Sources, "_series", stmt.Database),
		Condition:  rewriteSourcesCondition(stmt.Sources, stmt.Condition),
		Offset:     stmt.Offset,
		Limit:      stmt.Limit,
		SortFields: stmt.SortFields,
		OmitTime:   true,
		Dedupe:     true,
	}, nil
}

func rewriteShowTagValuesStatement(stmt *ShowTagValuesStatement) (Statement, error) {
	// Check for time in WHERE clause (not supported).
	if HasTimeExpr(stmt.Condition) {
		return nil, errors.New("SHOW TAG VALUES doesn't support time in WHERE clause")
	}

	condition := stmt.Condition
	var expr Expr
	if list, ok := stmt.TagKeyExpr.(*ListLiteral); ok {
		for _, tagKey := range list.Vals {
			tagExpr := &BinaryExpr{
				Op:  EQ,
				LHS: &VarRef{Val: "_tagKey"},
				RHS: &StringLiteral{Val: tagKey},
			}

			if expr != nil {
				expr = &BinaryExpr{
					Op:  OR,
					LHS: expr,
					RHS: tagExpr,
				}
			} else {
				expr = tagExpr
			}
		}
	} else {
		expr = &BinaryExpr{
			Op:  stmt.Op,
			LHS: &VarRef{Val: "_tagKey"},
			RHS: stmt.TagKeyExpr,
		}
	}

	// Set condition or "AND" together.
	if condition == nil {
		condition = expr
	} else {
		condition = &BinaryExpr{
			Op:  AND,
			LHS: &ParenExpr{Expr: condition},
			RHS: &ParenExpr{Expr: expr},
		}
	}
	condition = rewriteSourcesCondition(stmt.Sources, condition)

	return &ShowTagValuesStatement{
		Database:   stmt.Database,
		Op:         stmt.Op,
		TagKeyExpr: stmt.TagKeyExpr,
		Condition:  condition,
		SortFields: stmt.SortFields,
		Limit:      stmt.Limit,
		Offset:     stmt.Offset,
	}, nil
}

func rewriteShowTagKeysStatement(stmt *ShowTagKeysStatement) (Statement, error) {
	// Check for time in WHERE clause (not supported).
	if HasTimeExpr(stmt.Condition) {
		return nil, errors.New("SHOW TAG KEYS doesn't support time in WHERE clause")
	}

	return &SelectStatement{
		Fields: []*Field{
			{Expr: &VarRef{Val: "tagKey"}},
		},
		Sources:    rewriteSources(stmt.Sources, "_tagKeys", stmt.Database),
		Condition:  rewriteSourcesCondition(stmt.Sources, stmt.Condition),
		Offset:     stmt.Offset,
		Limit:      stmt.Limit,
		SortFields: stmt.SortFields,
		OmitTime:   true,
		Dedupe:     true,
	}, nil
}

// rewriteSources rewrites sources with previous database and retention policy
func rewriteSources(sources Sources, measurementName, defaultDatabase string) Sources {
	newSources := Sources{}
	for _, src := range sources {
		if src == nil {
			continue
		}
		mm := src.(*Measurement)
		database := mm.Database
		if database == "" {
			database = defaultDatabase
		}
		newSources = append(newSources,
			&Measurement{
				Database:        database,
				RetentionPolicy: mm.RetentionPolicy,
				Name:            measurementName,
			})
	}
	if len(newSources) <= 0 {
		return append(newSources, &Measurement{
			Database: defaultDatabase,
			Name:     measurementName,
		})
	}
	return newSources
}

// rewriteSourcesCondition rewrites sources into `name` expressions.
// Merges with cond and returns a new condition.
func rewriteSourcesCondition(sources Sources, cond Expr) Expr {
	if len(sources) == 0 {
		return cond
	}

	// Generate an OR'd set of filters on source name.
	var scond Expr
	for _, source := range sources {
		mm := source.(*Measurement)

		// Generate a filtering expression on the measurement name.
		var expr Expr
		if mm.Regex != nil {
			expr = &BinaryExpr{
				Op:  EQREGEX,
				LHS: &VarRef{Val: "_name"},
				RHS: &RegexLiteral{Val: mm.Regex.Val},
			}
		} else if mm.Name != "" {
			expr = &BinaryExpr{
				Op:  EQ,
				LHS: &VarRef{Val: "_name"},
				RHS: &StringLiteral{Val: mm.Name},
			}
		}

		if scond == nil {
			scond = expr
		} else {
			scond = &BinaryExpr{
				Op:  OR,
				LHS: scond,
				RHS: expr,
			}
		}
	}

	if cond != nil {
		return &BinaryExpr{
			Op:  AND,
			LHS: &ParenExpr{Expr: scond},
			RHS: &ParenExpr{Expr: cond},
		}
	}
	return scond
}
