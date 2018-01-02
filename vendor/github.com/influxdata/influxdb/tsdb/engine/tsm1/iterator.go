package tsm1

import (
	"fmt"

	"github.com/influxdata/influxdb/influxql"
)

func newLimitIterator(input influxql.Iterator, opt influxql.IteratorOptions) influxql.Iterator {
	switch input := input.(type) {
	case influxql.FloatIterator:
		return newFloatLimitIterator(input, opt)
	case influxql.IntegerIterator:
		return newIntegerLimitIterator(input, opt)
	case influxql.StringIterator:
		return newStringLimitIterator(input, opt)
	case influxql.BooleanIterator:
		return newBooleanLimitIterator(input, opt)
	default:
		panic(fmt.Sprintf("unsupported limit iterator type: %T", input))
	}
}

type floatCastIntegerCursor struct {
	cursor integerCursor
}

func (c *floatCastIntegerCursor) close() error { return c.cursor.close() }

func (c *floatCastIntegerCursor) next() (t int64, v interface{}) { return c.nextFloat() }

func (c *floatCastIntegerCursor) nextFloat() (int64, float64) {
	t, v := c.cursor.nextInteger()
	return t, float64(v)
}

type integerCastFloatCursor struct {
	cursor floatCursor
}

func (c *integerCastFloatCursor) close() error { return c.cursor.close() }

func (c *integerCastFloatCursor) next() (t int64, v interface{}) { return c.nextInteger() }

func (c *integerCastFloatCursor) nextInteger() (int64, int64) {
	t, v := c.cursor.nextFloat()
	return t, int64(v)
}
