package influxql

import (
	"errors"
	"fmt"
	"sort"
	"time"
)

// SelectOptions are options that customize the select call.
type SelectOptions struct {
	// The lower bound for a select call.
	MinTime time.Time

	// The upper bound for a select call.
	MaxTime time.Time

	// Node to exclusively read from.
	// If zero, all nodes are used.
	NodeID uint64

	// An optional channel that, if closed, signals that the select should be
	// interrupted.
	InterruptCh <-chan struct{}

	// Maximum number of concurrent series.
	MaxSeriesN int
}

// Select executes stmt against ic and returns a list of iterators to stream from.
//
// Statements should have all rewriting performed before calling select(). This
// includes wildcard and source expansion.
func Select(stmt *SelectStatement, ic IteratorCreator, sopt *SelectOptions) ([]Iterator, error) {
	// Determine base options for iterators.
	opt, err := newIteratorOptionsStmt(stmt, sopt)
	if err != nil {
		return nil, err
	}

	// Retrieve refs for each call and var ref.
	info := newSelectInfo(stmt)
	if len(info.calls) > 1 && len(info.refs) > 0 {
		return nil, errors.New("cannot select fields when selecting multiple aggregates")
	}

	// Determine auxiliary fields to be selected.
	opt.Aux = make([]VarRef, 0, len(info.refs))
	for ref := range info.refs {
		opt.Aux = append(opt.Aux, *ref)
	}
	sort.Sort(VarRefs(opt.Aux))

	// If there are multiple auxilary fields and no calls then construct an aux iterator.
	if len(info.calls) == 0 && len(info.refs) > 0 {
		return buildAuxIterators(stmt.Fields, ic, opt)
	}

	// Include auxiliary fields from top() and bottom()
	extraFields := 0
	for call := range info.calls {
		if call.Name == "top" || call.Name == "bottom" {
			for i := 1; i < len(call.Args)-1; i++ {
				ref := call.Args[i].(*VarRef)
				opt.Aux = append(opt.Aux, *ref)
				extraFields++
			}
		}
	}

	fields := stmt.Fields
	if extraFields > 0 {
		// Rebuild the list of fields if any extra fields are being implicitly added
		fields = make([]*Field, 0, len(stmt.Fields)+extraFields)
		for _, f := range stmt.Fields {
			fields = append(fields, f)
			switch expr := f.Expr.(type) {
			case *Call:
				if expr.Name == "top" || expr.Name == "bottom" {
					for i := 1; i < len(expr.Args)-1; i++ {
						fields = append(fields, &Field{Expr: expr.Args[i]})
					}
				}
			}
		}
	}

	// Determine if there is one call and it is a selector.
	selector := false
	if len(info.calls) == 1 {
		for call := range info.calls {
			switch call.Name {
			case "first", "last", "min", "max", "percentile":
				selector = true
			}
		}
	}

	return buildFieldIterators(fields, ic, opt, selector)
}

// buildAuxIterators creates a set of iterators from a single combined auxilary iterator.
func buildAuxIterators(fields Fields, ic IteratorCreator, opt IteratorOptions) ([]Iterator, error) {
	// Create iterator to read auxilary fields.
	input, err := ic.CreateIterator(opt)
	if err != nil {
		return nil, err
	} else if input == nil {
		input = &nilFloatIterator{}
	}

	// Filter out duplicate rows, if required.
	if opt.Dedupe {
		// If there is no group by and it is a float iterator, see if we can use a fast dedupe.
		if itr, ok := input.(FloatIterator); ok && len(opt.Dimensions) == 0 {
			if sz := len(fields); sz > 0 && sz < 3 {
				input = newFloatFastDedupeIterator(itr)
			} else {
				input = NewDedupeIterator(itr)
			}
		} else {
			input = NewDedupeIterator(input)
		}
	}

	// Apply limit & offset.
	if opt.Limit > 0 || opt.Offset > 0 {
		input = NewLimitIterator(input, opt)
	}

	// Wrap in an auxilary iterator to separate the fields.
	aitr := NewAuxIterator(input, opt)

	// Generate iterators for each field.
	itrs := make([]Iterator, len(fields))
	if err := func() error {
		for i, f := range fields {
			expr := Reduce(f.Expr, nil)
			switch expr := expr.(type) {
			case *VarRef:
				itrs[i] = aitr.Iterator(expr.Val, expr.Type)
			case *BinaryExpr:
				itr, err := buildExprIterator(expr, aitr, opt, false)
				if err != nil {
					return fmt.Errorf("error constructing iterator for field '%s': %s", f.String(), err)
				}
				itrs[i] = itr
			default:
				return fmt.Errorf("invalid expression type: %T", expr)
			}
		}
		return nil
	}(); err != nil {
		Iterators(Iterators(itrs).filterNonNil()).Close()
		aitr.Close()
		return nil, err
	}

	// Background the primary iterator since there is no reader for it.
	aitr.Background()

	return itrs, nil
}

// buildFieldIterators creates an iterator for each field expression.
func buildFieldIterators(fields Fields, ic IteratorCreator, opt IteratorOptions, selector bool) ([]Iterator, error) {
	// Create iterators from fields against the iterator creator.
	itrs := make([]Iterator, len(fields))

	if err := func() error {
		hasAuxFields := false

		var input Iterator
		for i, f := range fields {
			// Build iterators for calls first and save the iterator.
			// We do this so we can keep the ordering provided by the user, but
			// still build the Call's iterator first.
			if ContainsVarRef(f.Expr) {
				hasAuxFields = true
				continue
			}

			expr := Reduce(f.Expr, nil)
			itr, err := buildExprIterator(expr, ic, opt, selector)
			if err != nil {
				return err
			}
			itrs[i] = itr
			input = itr
		}

		if input == nil || !hasAuxFields {
			return nil
		}

		// Build the aux iterators. Previous validation should ensure that only one
		// call was present so we build an AuxIterator from that input.
		aitr := NewAuxIterator(input, opt)
		for i, f := range fields {
			if itrs[i] != nil {
				itrs[i] = aitr
				continue
			}

			expr := Reduce(f.Expr, nil)
			itr, err := buildExprIterator(expr, aitr, opt, false)
			if err != nil {
				return err
			}
			itrs[i] = itr
		}
		aitr.Start()
		return nil

	}(); err != nil {
		Iterators(Iterators(itrs).filterNonNil()).Close()
		return nil, err
	}

	// If there is a limit or offset then apply it.
	if opt.Limit > 0 || opt.Offset > 0 {
		for i := range itrs {
			itrs[i] = NewLimitIterator(itrs[i], opt)
		}
	}

	return itrs, nil
}

// buildExprIterator creates an iterator for an expression.
func buildExprIterator(expr Expr, ic IteratorCreator, opt IteratorOptions, selector bool) (Iterator, error) {
	opt.Expr = expr

	switch expr := expr.(type) {
	case *VarRef:
		itr, err := ic.CreateIterator(opt)
		if err != nil {
			return nil, err
		} else if itr == nil {
			itr = &nilFloatIterator{}
		}
		return itr, nil
	case *Call:
		// FIXME(benbjohnson): Validate that only calls with 1 arg are passed to IC.

		switch expr.Name {
		case "distinct":
			input, err := buildExprIterator(expr.Args[0].(*VarRef), ic, opt, selector)
			if err != nil {
				return nil, err
			}
			input, err = NewDistinctIterator(input, opt)
			if err != nil {
				return nil, err
			}
			return NewIntervalIterator(input, opt), nil
		case "sample":
			input, err := buildExprIterator(expr.Args[0], ic, opt, selector)
			if err != nil {
				return nil, err
			}
			size := expr.Args[1].(*IntegerLiteral)

			return newSampleIterator(input, opt, int(size.Val))
		case "holt_winters", "holt_winters_with_fit":
			input, err := buildExprIterator(expr.Args[0], ic, opt, selector)
			if err != nil {
				return nil, err
			}
			h := expr.Args[1].(*IntegerLiteral)
			m := expr.Args[2].(*IntegerLiteral)

			includeFitData := "holt_winters_with_fit" == expr.Name

			interval := opt.Interval.Duration
			// Redifine interval to be unbounded to capture all aggregate results
			opt.StartTime = MinTime
			opt.EndTime = MaxTime
			opt.Interval = Interval{}

			return newHoltWintersIterator(input, opt, int(h.Val), int(m.Val), includeFitData, interval)
		case "derivative", "non_negative_derivative", "difference", "moving_average", "elapsed":
			if !opt.Interval.IsZero() {
				if opt.Ascending {
					opt.StartTime -= int64(opt.Interval.Duration)
				} else {
					opt.EndTime += int64(opt.Interval.Duration)
				}
			}

			input, err := buildExprIterator(expr.Args[0], ic, opt, selector)
			if err != nil {
				return nil, err
			}

			switch expr.Name {
			case "derivative", "non_negative_derivative":
				interval := opt.DerivativeInterval()
				isNonNegative := (expr.Name == "non_negative_derivative")
				return newDerivativeIterator(input, opt, interval, isNonNegative)
			case "elapsed":
				interval := opt.ElapsedInterval()
				return newElapsedIterator(input, opt, interval)
			case "difference":
				return newDifferenceIterator(input, opt)
			case "moving_average":
				n := expr.Args[1].(*IntegerLiteral)
				if n.Val > 1 && !opt.Interval.IsZero() {
					if opt.Ascending {
						opt.StartTime -= int64(opt.Interval.Duration) * (n.Val - 1)
					} else {
						opt.EndTime += int64(opt.Interval.Duration) * (n.Val - 1)
					}
				}
				return newMovingAverageIterator(input, int(n.Val), opt)
			}
			panic(fmt.Sprintf("invalid series aggregate function: %s", expr.Name))
		case "cumulative_sum":
			input, err := buildExprIterator(expr.Args[0], ic, opt, selector)
			if err != nil {
				return nil, err
			}
			return newCumulativeSumIterator(input, opt)
		default:
			itr, err := func() (Iterator, error) {
				switch expr.Name {
				case "count":
					switch arg := expr.Args[0].(type) {
					case *Call:
						if arg.Name == "distinct" {
							input, err := buildExprIterator(arg, ic, opt, selector)
							if err != nil {
								return nil, err
							}
							return newCountIterator(input, opt)
						}
					}

					itr, err := ic.CreateIterator(opt)
					if err != nil {
						return nil, err
					} else if itr == nil {
						itr = &nilFloatIterator{}
					}
					return itr, nil
				case "min", "max", "sum", "first", "last", "mean":
					itr, err := ic.CreateIterator(opt)
					if err != nil {
						return nil, err
					} else if itr == nil {
						itr = &nilFloatIterator{}
					}
					return itr, nil
				case "median":
					input, err := buildExprIterator(expr.Args[0].(*VarRef), ic, opt, false)
					if err != nil {
						return nil, err
					}
					return newMedianIterator(input, opt)
				case "mode":
					input, err := buildExprIterator(expr.Args[0].(*VarRef), ic, opt, false)
					if err != nil {
						return nil, err
					}
					return NewModeIterator(input, opt)
				case "stddev":
					input, err := buildExprIterator(expr.Args[0].(*VarRef), ic, opt, false)
					if err != nil {
						return nil, err
					}
					return newStddevIterator(input, opt)
				case "spread":
					// OPTIMIZE(benbjohnson): convert to map/reduce
					input, err := buildExprIterator(expr.Args[0].(*VarRef), ic, opt, false)
					if err != nil {
						return nil, err
					}
					return newSpreadIterator(input, opt)
				case "top":
					var tags []int
					if len(expr.Args) < 2 {
						return nil, fmt.Errorf("top() requires 2 or more arguments, got %d", len(expr.Args))
					} else if len(expr.Args) > 2 {
						// We need to find the indices of where the tag values are stored in Aux
						// This section is O(n^2), but for what should be a low value.
						for i := 1; i < len(expr.Args)-1; i++ {
							ref := expr.Args[i].(*VarRef)
							for index, aux := range opt.Aux {
								if aux.Val == ref.Val {
									tags = append(tags, index)
									break
								}
							}
						}
					}

					input, err := buildExprIterator(expr.Args[0].(*VarRef), ic, opt, false)
					if err != nil {
						return nil, err
					}
					n := expr.Args[len(expr.Args)-1].(*IntegerLiteral)
					return newTopIterator(input, opt, n, tags)
				case "bottom":
					var tags []int
					if len(expr.Args) < 2 {
						return nil, fmt.Errorf("bottom() requires 2 or more arguments, got %d", len(expr.Args))
					} else if len(expr.Args) > 2 {
						// We need to find the indices of where the tag values are stored in Aux
						// This section is O(n^2), but for what should be a low value.
						for i := 1; i < len(expr.Args)-1; i++ {
							ref := expr.Args[i].(*VarRef)
							for index, aux := range opt.Aux {
								if aux.Val == ref.Val {
									tags = append(tags, index)
									break
								}
							}
						}
					}

					input, err := buildExprIterator(expr.Args[0].(*VarRef), ic, opt, false)
					if err != nil {
						return nil, err
					}
					n := expr.Args[len(expr.Args)-1].(*IntegerLiteral)
					return newBottomIterator(input, opt, n, tags)
				case "percentile":
					input, err := buildExprIterator(expr.Args[0].(*VarRef), ic, opt, false)
					if err != nil {
						return nil, err
					}
					var percentile float64
					switch arg := expr.Args[1].(type) {
					case *NumberLiteral:
						percentile = arg.Val
					case *IntegerLiteral:
						percentile = float64(arg.Val)
					}
					return newPercentileIterator(input, opt, percentile)
				default:
					return nil, fmt.Errorf("unsupported call: %s", expr.Name)
				}
			}()

			if err != nil {
				return nil, err
			}

			if !selector || !opt.Interval.IsZero() {
				if expr.Name != "top" && expr.Name != "bottom" {
					itr = NewIntervalIterator(itr, opt)
				}
				if !opt.Interval.IsZero() && opt.Fill != NoFill {
					itr = NewFillIterator(itr, expr, opt)
				}
			}
			if opt.InterruptCh != nil {
				itr = NewInterruptIterator(itr, opt.InterruptCh)
			}
			return itr, nil
		}
	case *BinaryExpr:
		if rhs, ok := expr.RHS.(Literal); ok {
			// The right hand side is a literal. It is more common to have the RHS be a literal,
			// so we check that one first and have this be the happy path.
			if lhs, ok := expr.LHS.(Literal); ok {
				// We have two literals that couldn't be combined by Reduce.
				return nil, fmt.Errorf("unable to construct an iterator from two literals: LHS: %T, RHS: %T", lhs, rhs)
			}

			lhs, err := buildExprIterator(expr.LHS, ic, opt, false)
			if err != nil {
				return nil, err
			}
			return buildRHSTransformIterator(lhs, rhs, expr.Op, ic, opt)
		} else if lhs, ok := expr.LHS.(Literal); ok {
			rhs, err := buildExprIterator(expr.RHS, ic, opt, false)
			if err != nil {
				return nil, err
			}
			return buildLHSTransformIterator(lhs, rhs, expr.Op, ic, opt)
		} else {
			// We have two iterators. Combine them into a single iterator.
			lhs, err := buildExprIterator(expr.LHS, ic, opt, false)
			if err != nil {
				return nil, err
			}
			rhs, err := buildExprIterator(expr.RHS, ic, opt, false)
			if err != nil {
				return nil, err
			}
			return buildTransformIterator(lhs, rhs, expr.Op, ic, opt)
		}
	case *ParenExpr:
		return buildExprIterator(expr.Expr, ic, opt, selector)
	default:
		return nil, fmt.Errorf("invalid expression type: %T", expr)
	}
}

func buildRHSTransformIterator(lhs Iterator, rhs Literal, op Token, ic IteratorCreator, opt IteratorOptions) (Iterator, error) {
	fn := binaryExprFunc(iteratorDataType(lhs), literalDataType(rhs), op)
	switch fn := fn.(type) {
	case func(float64, float64) float64:
		var input FloatIterator
		switch lhs := lhs.(type) {
		case FloatIterator:
			input = lhs
		case IntegerIterator:
			input = &integerFloatCastIterator{input: lhs}
		default:
			return nil, fmt.Errorf("type mismatch on LHS, unable to use %T as a FloatIterator", lhs)
		}

		var val float64
		switch rhs := rhs.(type) {
		case *NumberLiteral:
			val = rhs.Val
		case *IntegerLiteral:
			val = float64(rhs.Val)
		default:
			return nil, fmt.Errorf("type mismatch on RHS, unable to use %T as a NumberLiteral", rhs)
		}
		return &floatTransformIterator{
			input: input,
			fn: func(p *FloatPoint) *FloatPoint {
				if p == nil {
					return nil
				} else if p.Nil {
					return p
				}
				p.Value = fn(p.Value, val)
				return p
			},
		}, nil
	case func(int64, int64) float64:
		input, ok := lhs.(IntegerIterator)
		if !ok {
			return nil, fmt.Errorf("type mismatch on LHS, unable to use %T as a IntegerIterator", lhs)
		}

		var val int64
		switch rhs := rhs.(type) {
		case *IntegerLiteral:
			val = rhs.Val
		default:
			return nil, fmt.Errorf("type mismatch on RHS, unable to use %T as a IntegerLiteral", rhs)
		}
		return &integerFloatTransformIterator{
			input: input,
			fn: func(p *IntegerPoint) *FloatPoint {
				if p == nil {
					return nil
				}

				fp := &FloatPoint{
					Name: p.Name,
					Tags: p.Tags,
					Time: p.Time,
					Aux:  p.Aux,
				}
				if p.Nil {
					fp.Nil = true
				} else {
					fp.Value = fn(p.Value, val)
				}
				return fp
			},
		}, nil
	case func(float64, float64) bool:
		var input FloatIterator
		switch lhs := lhs.(type) {
		case FloatIterator:
			input = lhs
		case IntegerIterator:
			input = &integerFloatCastIterator{input: lhs}
		default:
			return nil, fmt.Errorf("type mismatch on LHS, unable to use %T as a FloatIterator", lhs)
		}

		var val float64
		switch rhs := rhs.(type) {
		case *NumberLiteral:
			val = rhs.Val
		case *IntegerLiteral:
			val = float64(rhs.Val)
		default:
			return nil, fmt.Errorf("type mismatch on RHS, unable to use %T as a NumberLiteral", rhs)
		}
		return &floatBoolTransformIterator{
			input: input,
			fn: func(p *FloatPoint) *BooleanPoint {
				if p == nil {
					return nil
				}

				bp := &BooleanPoint{
					Name: p.Name,
					Tags: p.Tags,
					Time: p.Time,
					Aux:  p.Aux,
				}
				if p.Nil {
					bp.Nil = true
				} else {
					bp.Value = fn(p.Value, val)
				}
				return bp
			},
		}, nil
	case func(int64, int64) int64:
		var input IntegerIterator
		switch lhs := lhs.(type) {
		case IntegerIterator:
			input = lhs
		default:
			return nil, fmt.Errorf("type mismatch on LHS, unable to use %T as an IntegerIterator", lhs)
		}

		var val int64
		switch rhs := rhs.(type) {
		case *IntegerLiteral:
			val = rhs.Val
		default:
			return nil, fmt.Errorf("type mismatch on RHS, unable to use %T as an IntegerLiteral", rhs)
		}
		return &integerTransformIterator{
			input: input,
			fn: func(p *IntegerPoint) *IntegerPoint {
				if p == nil {
					return nil
				} else if p.Nil {
					return p
				}
				p.Value = fn(p.Value, val)
				return p
			},
		}, nil
	case func(int64, int64) bool:
		var input IntegerIterator
		switch lhs := lhs.(type) {
		case IntegerIterator:
			input = lhs
		default:
			return nil, fmt.Errorf("type mismatch on LHS, unable to use %T as an IntegerIterator", lhs)
		}

		var val int64
		switch rhs := rhs.(type) {
		case *IntegerLiteral:
			val = rhs.Val
		default:
			return nil, fmt.Errorf("type mismatch on RHS, unable to use %T as an IntegerLiteral", rhs)
		}
		return &integerBoolTransformIterator{
			input: input,
			fn: func(p *IntegerPoint) *BooleanPoint {
				if p == nil {
					return nil
				}

				bp := &BooleanPoint{
					Name: p.Name,
					Tags: p.Tags,
					Time: p.Time,
					Aux:  p.Aux,
				}
				if p.Nil {
					bp.Nil = true
				} else {
					bp.Value = fn(p.Value, val)
				}
				return bp
			},
		}, nil
	}
	return nil, fmt.Errorf("unable to construct rhs transform iterator from %T and %T", lhs, rhs)
}

func buildLHSTransformIterator(lhs Literal, rhs Iterator, op Token, ic IteratorCreator, opt IteratorOptions) (Iterator, error) {
	fn := binaryExprFunc(literalDataType(lhs), iteratorDataType(rhs), op)
	switch fn := fn.(type) {
	case func(float64, float64) float64:
		var input FloatIterator
		switch rhs := rhs.(type) {
		case FloatIterator:
			input = rhs
		case IntegerIterator:
			input = &integerFloatCastIterator{input: rhs}
		default:
			return nil, fmt.Errorf("type mismatch on RHS, unable to use %T as a FloatIterator", rhs)
		}

		var val float64
		switch lhs := lhs.(type) {
		case *NumberLiteral:
			val = lhs.Val
		case *IntegerLiteral:
			val = float64(lhs.Val)
		default:
			return nil, fmt.Errorf("type mismatch on LHS, unable to use %T as a NumberLiteral", lhs)
		}
		return &floatTransformIterator{
			input: input,
			fn: func(p *FloatPoint) *FloatPoint {
				if p == nil {
					return nil
				} else if p.Nil {
					return p
				}
				p.Value = fn(val, p.Value)
				return p
			},
		}, nil
	case func(int64, int64) float64:
		input, ok := rhs.(IntegerIterator)
		if !ok {
			return nil, fmt.Errorf("type mismatch on RHS, unable to use %T as a IntegerIterator", lhs)
		}

		var val int64
		switch lhs := lhs.(type) {
		case *IntegerLiteral:
			val = lhs.Val
		default:
			return nil, fmt.Errorf("type mismatch on LHS, unable to use %T as a IntegerLiteral", rhs)
		}
		return &integerFloatTransformIterator{
			input: input,
			fn: func(p *IntegerPoint) *FloatPoint {
				if p == nil {
					return nil
				}

				fp := &FloatPoint{
					Name: p.Name,
					Tags: p.Tags,
					Time: p.Time,
					Aux:  p.Aux,
				}
				if p.Nil {
					fp.Nil = true
				} else {
					fp.Value = fn(val, p.Value)
				}
				return fp
			},
		}, nil
	case func(float64, float64) bool:
		var input FloatIterator
		switch rhs := rhs.(type) {
		case FloatIterator:
			input = rhs
		case IntegerIterator:
			input = &integerFloatCastIterator{input: rhs}
		default:
			return nil, fmt.Errorf("type mismatch on RHS, unable to use %T as a FloatIterator", rhs)
		}

		var val float64
		switch lhs := lhs.(type) {
		case *NumberLiteral:
			val = lhs.Val
		case *IntegerLiteral:
			val = float64(lhs.Val)
		default:
			return nil, fmt.Errorf("type mismatch on LHS, unable to use %T as a NumberLiteral", lhs)
		}
		return &floatBoolTransformIterator{
			input: input,
			fn: func(p *FloatPoint) *BooleanPoint {
				if p == nil {
					return nil
				}

				bp := &BooleanPoint{
					Name: p.Name,
					Tags: p.Tags,
					Time: p.Time,
					Aux:  p.Aux,
				}
				if p.Nil {
					bp.Nil = true
				} else {
					bp.Value = fn(val, p.Value)
				}
				return bp
			},
		}, nil
	case func(int64, int64) int64:
		var input IntegerIterator
		switch rhs := rhs.(type) {
		case IntegerIterator:
			input = rhs
		default:
			return nil, fmt.Errorf("type mismatch on RHS, unable to use %T as an IntegerIterator", rhs)
		}

		var val int64
		switch lhs := lhs.(type) {
		case *IntegerLiteral:
			val = lhs.Val
		default:
			return nil, fmt.Errorf("type mismatch on LHS, unable to use %T as an IntegerLiteral", lhs)
		}
		return &integerTransformIterator{
			input: input,
			fn: func(p *IntegerPoint) *IntegerPoint {
				if p == nil {
					return nil
				} else if p.Nil {
					return p
				}
				p.Value = fn(val, p.Value)
				return p
			},
		}, nil
	case func(int64, int64) bool:
		var input IntegerIterator
		switch rhs := rhs.(type) {
		case IntegerIterator:
			input = rhs
		default:
			return nil, fmt.Errorf("type mismatch on RHS, unable to use %T as an IntegerIterator", rhs)
		}

		var val int64
		switch lhs := lhs.(type) {
		case *IntegerLiteral:
			val = lhs.Val
		default:
			return nil, fmt.Errorf("type mismatch on LHS, unable to use %T as an IntegerLiteral", lhs)
		}
		return &integerBoolTransformIterator{
			input: input,
			fn: func(p *IntegerPoint) *BooleanPoint {
				if p == nil {
					return nil
				}

				bp := &BooleanPoint{
					Name: p.Name,
					Tags: p.Tags,
					Time: p.Time,
					Aux:  p.Aux,
				}
				if p.Nil {
					bp.Nil = true
				} else {
					bp.Value = fn(val, p.Value)
				}
				return bp
			},
		}, nil
	}
	return nil, fmt.Errorf("unable to construct lhs transform iterator from %T and %T", lhs, rhs)
}

func buildTransformIterator(lhs Iterator, rhs Iterator, op Token, ic IteratorCreator, opt IteratorOptions) (Iterator, error) {
	fn := binaryExprFunc(iteratorDataType(lhs), iteratorDataType(rhs), op)
	switch fn := fn.(type) {
	case func(float64, float64) float64:
		var left FloatIterator
		switch lhs := lhs.(type) {
		case FloatIterator:
			left = lhs
		case IntegerIterator:
			left = &integerFloatCastIterator{input: lhs}
		default:
			return nil, fmt.Errorf("type mismatch on LHS, unable to use %T as a FloatIterator", lhs)
		}

		var right FloatIterator
		switch rhs := rhs.(type) {
		case FloatIterator:
			right = rhs
		case IntegerIterator:
			right = &integerFloatCastIterator{input: rhs}
		default:
			return nil, fmt.Errorf("type mismatch on RHS, unable to use %T as a FloatIterator", rhs)
		}
		return newFloatExprIterator(left, right, opt, fn), nil
	case func(int64, int64) float64:
		left, ok := lhs.(IntegerIterator)
		if !ok {
			return nil, fmt.Errorf("type mismatch on LHS, unable to use %T as a IntegerIterator", lhs)
		}
		right, ok := rhs.(IntegerIterator)
		if !ok {
			return nil, fmt.Errorf("type mismatch on RHS, unable to use %T as a IntegerIterator", rhs)
		}
		return newIntegerFloatExprIterator(left, right, opt, fn), nil
	case func(int64, int64) int64:
		left, ok := lhs.(IntegerIterator)
		if !ok {
			return nil, fmt.Errorf("type mismatch on LHS, unable to use %T as a IntegerIterator", lhs)
		}
		right, ok := rhs.(IntegerIterator)
		if !ok {
			return nil, fmt.Errorf("type mismatch on RHS, unable to use %T as a IntegerIterator", rhs)
		}
		return newIntegerExprIterator(left, right, opt, fn), nil
	case func(float64, float64) bool:
		var left FloatIterator
		switch lhs := lhs.(type) {
		case FloatIterator:
			left = lhs
		case IntegerIterator:
			left = &integerFloatCastIterator{input: lhs}
		default:
			return nil, fmt.Errorf("type mismatch on LHS, unable to use %T as a FloatIterator", lhs)
		}

		var right FloatIterator
		switch rhs := rhs.(type) {
		case FloatIterator:
			right = rhs
		case IntegerIterator:
			right = &integerFloatCastIterator{input: rhs}
		default:
			return nil, fmt.Errorf("type mismatch on RHS, unable to use %T as a FloatIterator", rhs)
		}
		return newFloatBooleanExprIterator(left, right, opt, fn), nil
	case func(int64, int64) bool:
		left, ok := lhs.(IntegerIterator)
		if !ok {
			return nil, fmt.Errorf("type mismatch on LHS, unable to use %T as a IntegerIterator", lhs)
		}
		right, ok := rhs.(IntegerIterator)
		if !ok {
			return nil, fmt.Errorf("type mismatch on LHS, unable to use %T as a IntegerIterator", rhs)
		}
		return newIntegerBooleanExprIterator(left, right, opt, fn), nil
	}
	return nil, fmt.Errorf("unable to construct transform iterator from %T and %T", lhs, rhs)
}

func iteratorDataType(itr Iterator) DataType {
	switch itr.(type) {
	case FloatIterator:
		return Float
	case IntegerIterator:
		return Integer
	case StringIterator:
		return String
	case BooleanIterator:
		return Boolean
	default:
		return Unknown
	}
}

func literalDataType(lit Literal) DataType {
	switch lit.(type) {
	case *NumberLiteral:
		return Float
	case *IntegerLiteral:
		return Integer
	case *StringLiteral:
		return String
	case *BooleanLiteral:
		return Boolean
	default:
		return Unknown
	}
}

func binaryExprFunc(typ1 DataType, typ2 DataType, op Token) interface{} {
	var fn interface{}
	switch typ1 {
	case Float:
		fn = floatBinaryExprFunc(op)
	case Integer:
		switch typ2 {
		case Float:
			fn = floatBinaryExprFunc(op)
		default:
			fn = integerBinaryExprFunc(op)
		}
	}
	return fn
}

func floatBinaryExprFunc(op Token) interface{} {
	switch op {
	case ADD:
		return func(lhs, rhs float64) float64 { return lhs + rhs }
	case SUB:
		return func(lhs, rhs float64) float64 { return lhs - rhs }
	case MUL:
		return func(lhs, rhs float64) float64 { return lhs * rhs }
	case DIV:
		return func(lhs, rhs float64) float64 {
			if rhs == 0 {
				return float64(0)
			}
			return lhs / rhs
		}
	case EQ:
		return func(lhs, rhs float64) bool { return lhs == rhs }
	case NEQ:
		return func(lhs, rhs float64) bool { return lhs != rhs }
	case LT:
		return func(lhs, rhs float64) bool { return lhs < rhs }
	case LTE:
		return func(lhs, rhs float64) bool { return lhs <= rhs }
	case GT:
		return func(lhs, rhs float64) bool { return lhs > rhs }
	case GTE:
		return func(lhs, rhs float64) bool { return lhs >= rhs }
	}
	return nil
}

func integerBinaryExprFunc(op Token) interface{} {
	switch op {
	case ADD:
		return func(lhs, rhs int64) int64 { return lhs + rhs }
	case SUB:
		return func(lhs, rhs int64) int64 { return lhs - rhs }
	case MUL:
		return func(lhs, rhs int64) int64 { return lhs * rhs }
	case DIV:
		return func(lhs, rhs int64) float64 {
			if rhs == 0 {
				return float64(0)
			}
			return float64(lhs) / float64(rhs)
		}
	case EQ:
		return func(lhs, rhs int64) bool { return lhs == rhs }
	case NEQ:
		return func(lhs, rhs int64) bool { return lhs != rhs }
	case LT:
		return func(lhs, rhs int64) bool { return lhs < rhs }
	case LTE:
		return func(lhs, rhs int64) bool { return lhs <= rhs }
	case GT:
		return func(lhs, rhs int64) bool { return lhs > rhs }
	case GTE:
		return func(lhs, rhs int64) bool { return lhs >= rhs }
	}
	return nil
}

// stringSetSlice returns a sorted slice of keys from a string set.
func stringSetSlice(m map[string]struct{}) []string {
	if m == nil {
		return nil
	}

	a := make([]string, 0, len(m))
	for k := range m {
		a = append(a, k)
	}
	sort.Strings(a)
	return a
}
