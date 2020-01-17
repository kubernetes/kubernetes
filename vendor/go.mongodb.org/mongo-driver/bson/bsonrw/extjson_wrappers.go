// Copyright (C) MongoDB, Inc. 2017-present.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

package bsonrw

import (
	"encoding/base64"
	"errors"
	"fmt"
	"math"
	"strconv"
	"time"

	"go.mongodb.org/mongo-driver/bson/bsontype"
	"go.mongodb.org/mongo-driver/bson/primitive"
)

func wrapperKeyBSONType(key string) bsontype.Type {
	switch string(key) {
	case "$numberInt":
		return bsontype.Int32
	case "$numberLong":
		return bsontype.Int64
	case "$oid":
		return bsontype.ObjectID
	case "$symbol":
		return bsontype.Symbol
	case "$numberDouble":
		return bsontype.Double
	case "$numberDecimal":
		return bsontype.Decimal128
	case "$binary":
		return bsontype.Binary
	case "$code":
		return bsontype.JavaScript
	case "$scope":
		return bsontype.CodeWithScope
	case "$timestamp":
		return bsontype.Timestamp
	case "$regularExpression":
		return bsontype.Regex
	case "$dbPointer":
		return bsontype.DBPointer
	case "$date":
		return bsontype.DateTime
	case "$ref":
		fallthrough
	case "$id":
		fallthrough
	case "$db":
		return bsontype.EmbeddedDocument // dbrefs aren't bson types
	case "$minKey":
		return bsontype.MinKey
	case "$maxKey":
		return bsontype.MaxKey
	case "$undefined":
		return bsontype.Undefined
	}

	return bsontype.EmbeddedDocument
}

func (ejv *extJSONValue) parseBinary() (b []byte, subType byte, err error) {
	if ejv.t != bsontype.EmbeddedDocument {
		return nil, 0, fmt.Errorf("$binary value should be object, but instead is %s", ejv.t)
	}

	binObj := ejv.v.(*extJSONObject)
	bFound := false
	stFound := false

	for i, key := range binObj.keys {
		val := binObj.values[i]

		switch key {
		case "base64":
			if bFound {
				return nil, 0, errors.New("duplicate base64 key in $binary")
			}

			if val.t != bsontype.String {
				return nil, 0, fmt.Errorf("$binary base64 value should be string, but instead is %s", val.t)
			}

			base64Bytes, err := base64.StdEncoding.DecodeString(val.v.(string))
			if err != nil {
				return nil, 0, fmt.Errorf("invalid $binary base64 string: %s", val.v.(string))
			}

			b = base64Bytes
			bFound = true
		case "subType":
			if stFound {
				return nil, 0, errors.New("duplicate subType key in $binary")
			}

			if val.t != bsontype.String {
				return nil, 0, fmt.Errorf("$binary subType value should be string, but instead is %s", val.t)
			}

			i, err := strconv.ParseInt(val.v.(string), 16, 64)
			if err != nil {
				return nil, 0, fmt.Errorf("invalid $binary subType string: %s", val.v.(string))
			}

			subType = byte(i)
			stFound = true
		default:
			return nil, 0, fmt.Errorf("invalid key in $binary object: %s", key)
		}
	}

	if !bFound {
		return nil, 0, errors.New("missing base64 field in $binary object")
	}

	if !stFound {
		return nil, 0, errors.New("missing subType field in $binary object")

	}

	return b, subType, nil
}

func (ejv *extJSONValue) parseDBPointer() (ns string, oid primitive.ObjectID, err error) {
	if ejv.t != bsontype.EmbeddedDocument {
		return "", primitive.NilObjectID, fmt.Errorf("$dbPointer value should be object, but instead is %s", ejv.t)
	}

	dbpObj := ejv.v.(*extJSONObject)
	oidFound := false
	nsFound := false

	for i, key := range dbpObj.keys {
		val := dbpObj.values[i]

		switch key {
		case "$ref":
			if nsFound {
				return "", primitive.NilObjectID, errors.New("duplicate $ref key in $dbPointer")
			}

			if val.t != bsontype.String {
				return "", primitive.NilObjectID, fmt.Errorf("$dbPointer $ref value should be string, but instead is %s", val.t)
			}

			ns = val.v.(string)
			nsFound = true
		case "$id":
			if oidFound {
				return "", primitive.NilObjectID, errors.New("duplicate $id key in $dbPointer")
			}

			if val.t != bsontype.String {
				return "", primitive.NilObjectID, fmt.Errorf("$dbPointer $id value should be string, but instead is %s", val.t)
			}

			oid, err = primitive.ObjectIDFromHex(val.v.(string))
			if err != nil {
				return "", primitive.NilObjectID, err
			}

			oidFound = true
		default:
			return "", primitive.NilObjectID, fmt.Errorf("invalid key in $dbPointer object: %s", key)
		}
	}

	if !nsFound {
		return "", oid, errors.New("missing $ref field in $dbPointer object")
	}

	if !oidFound {
		return "", oid, errors.New("missing $id field in $dbPointer object")
	}

	return ns, oid, nil
}

const rfc3339Milli = "2006-01-02T15:04:05.999Z07:00"

func (ejv *extJSONValue) parseDateTime() (int64, error) {
	switch ejv.t {
	case bsontype.Int32:
		return int64(ejv.v.(int32)), nil
	case bsontype.Int64:
		return ejv.v.(int64), nil
	case bsontype.String:
		return parseDatetimeString(ejv.v.(string))
	case bsontype.EmbeddedDocument:
		return parseDatetimeObject(ejv.v.(*extJSONObject))
	default:
		return 0, fmt.Errorf("$date value should be string or object, but instead is %s", ejv.t)
	}
}

func parseDatetimeString(data string) (int64, error) {
	t, err := time.Parse(rfc3339Milli, data)
	if err != nil {
		return 0, fmt.Errorf("invalid $date value string: %s", data)
	}

	return t.Unix()*1e3 + int64(t.Nanosecond())/1e6, nil
}

func parseDatetimeObject(data *extJSONObject) (d int64, err error) {
	dFound := false

	for i, key := range data.keys {
		val := data.values[i]

		switch key {
		case "$numberLong":
			if dFound {
				return 0, errors.New("duplicate $numberLong key in $date")
			}

			if val.t != bsontype.String {
				return 0, fmt.Errorf("$date $numberLong field should be string, but instead is %s", val.t)
			}

			d, err = val.parseInt64()
			if err != nil {
				return 0, err
			}
			dFound = true
		default:
			return 0, fmt.Errorf("invalid key in $date object: %s", key)
		}
	}

	if !dFound {
		return 0, errors.New("missing $numberLong field in $date object")
	}

	return d, nil
}

func (ejv *extJSONValue) parseDecimal128() (primitive.Decimal128, error) {
	if ejv.t != bsontype.String {
		return primitive.Decimal128{}, fmt.Errorf("$numberDecimal value should be string, but instead is %s", ejv.t)
	}

	d, err := primitive.ParseDecimal128(ejv.v.(string))
	if err != nil {
		return primitive.Decimal128{}, fmt.Errorf("$invalid $numberDecimal string: %s", ejv.v.(string))
	}

	return d, nil
}

func (ejv *extJSONValue) parseDouble() (float64, error) {
	if ejv.t == bsontype.Double {
		return ejv.v.(float64), nil
	}

	if ejv.t != bsontype.String {
		return 0, fmt.Errorf("$numberDouble value should be string, but instead is %s", ejv.t)
	}

	switch string(ejv.v.(string)) {
	case "Infinity":
		return math.Inf(1), nil
	case "-Infinity":
		return math.Inf(-1), nil
	case "NaN":
		return math.NaN(), nil
	}

	f, err := strconv.ParseFloat(ejv.v.(string), 64)
	if err != nil {
		return 0, err
	}

	return f, nil
}

func (ejv *extJSONValue) parseInt32() (int32, error) {
	if ejv.t == bsontype.Int32 {
		return ejv.v.(int32), nil
	}

	if ejv.t != bsontype.String {
		return 0, fmt.Errorf("$numberInt value should be string, but instead is %s", ejv.t)
	}

	i, err := strconv.ParseInt(ejv.v.(string), 10, 64)
	if err != nil {
		return 0, err
	}

	if i < math.MinInt32 || i > math.MaxInt32 {
		return 0, fmt.Errorf("$numberInt value should be int32 but instead is int64: %d", i)
	}

	return int32(i), nil
}

func (ejv *extJSONValue) parseInt64() (int64, error) {
	if ejv.t == bsontype.Int64 {
		return ejv.v.(int64), nil
	}

	if ejv.t != bsontype.String {
		return 0, fmt.Errorf("$numberLong value should be string, but instead is %s", ejv.t)
	}

	i, err := strconv.ParseInt(ejv.v.(string), 10, 64)
	if err != nil {
		return 0, err
	}

	return i, nil
}

func (ejv *extJSONValue) parseJavascript() (code string, err error) {
	if ejv.t != bsontype.String {
		return "", fmt.Errorf("$code value should be string, but instead is %s", ejv.t)
	}

	return ejv.v.(string), nil
}

func (ejv *extJSONValue) parseMinMaxKey(minmax string) error {
	if ejv.t != bsontype.Int32 {
		return fmt.Errorf("$%sKey value should be int32, but instead is %s", minmax, ejv.t)
	}

	if ejv.v.(int32) != 1 {
		return fmt.Errorf("$%sKey value must be 1, but instead is %d", minmax, ejv.v.(int32))
	}

	return nil
}

func (ejv *extJSONValue) parseObjectID() (primitive.ObjectID, error) {
	if ejv.t != bsontype.String {
		return primitive.NilObjectID, fmt.Errorf("$oid value should be string, but instead is %s", ejv.t)
	}

	return primitive.ObjectIDFromHex(ejv.v.(string))
}

func (ejv *extJSONValue) parseRegex() (pattern, options string, err error) {
	if ejv.t != bsontype.EmbeddedDocument {
		return "", "", fmt.Errorf("$regularExpression value should be object, but instead is %s", ejv.t)
	}

	regexObj := ejv.v.(*extJSONObject)
	patFound := false
	optFound := false

	for i, key := range regexObj.keys {
		val := regexObj.values[i]

		switch string(key) {
		case "pattern":
			if patFound {
				return "", "", errors.New("duplicate pattern key in $regularExpression")
			}

			if val.t != bsontype.String {
				return "", "", fmt.Errorf("$regularExpression pattern value should be string, but instead is %s", val.t)
			}

			pattern = val.v.(string)
			patFound = true
		case "options":
			if optFound {
				return "", "", errors.New("duplicate options key in $regularExpression")
			}

			if val.t != bsontype.String {
				return "", "", fmt.Errorf("$regularExpression options value should be string, but instead is %s", val.t)
			}

			options = val.v.(string)
			optFound = true
		default:
			return "", "", fmt.Errorf("invalid key in $regularExpression object: %s", key)
		}
	}

	if !patFound {
		return "", "", errors.New("missing pattern field in $regularExpression object")
	}

	if !optFound {
		return "", "", errors.New("missing options field in $regularExpression object")

	}

	return pattern, options, nil
}

func (ejv *extJSONValue) parseSymbol() (string, error) {
	if ejv.t != bsontype.String {
		return "", fmt.Errorf("$symbol value should be string, but instead is %s", ejv.t)
	}

	return ejv.v.(string), nil
}

func (ejv *extJSONValue) parseTimestamp() (t, i uint32, err error) {
	if ejv.t != bsontype.EmbeddedDocument {
		return 0, 0, fmt.Errorf("$timestamp value should be object, but instead is %s", ejv.t)
	}

	handleKey := func(key string, val *extJSONValue, flag bool) (uint32, error) {
		if flag {
			return 0, fmt.Errorf("duplicate %s key in $timestamp", key)
		}

		switch val.t {
		case bsontype.Int32:
			if val.v.(int32) < 0 {
				return 0, fmt.Errorf("$timestamp %s number should be uint32: %s", key, string(val.v.(int32)))
			}

			return uint32(val.v.(int32)), nil
		case bsontype.Int64:
			if val.v.(int64) < 0 || uint32(val.v.(int64)) > math.MaxUint32 {
				return 0, fmt.Errorf("$timestamp %s number should be uint32: %s", key, string(val.v.(int32)))
			}

			return uint32(val.v.(int64)), nil
		default:
			return 0, fmt.Errorf("$timestamp %s value should be uint32, but instead is %s", key, val.t)
		}
	}

	tsObj := ejv.v.(*extJSONObject)
	tFound := false
	iFound := false

	for j, key := range tsObj.keys {
		val := tsObj.values[j]

		switch key {
		case "t":
			if t, err = handleKey(key, val, tFound); err != nil {
				return 0, 0, err
			}

			tFound = true
		case "i":
			if i, err = handleKey(key, val, iFound); err != nil {
				return 0, 0, err
			}

			iFound = true
		default:
			return 0, 0, fmt.Errorf("invalid key in $timestamp object: %s", key)
		}
	}

	if !tFound {
		return 0, 0, errors.New("missing t field in $timestamp object")
	}

	if !iFound {
		return 0, 0, errors.New("missing i field in $timestamp object")
	}

	return t, i, nil
}

func (ejv *extJSONValue) parseUndefined() error {
	if ejv.t != bsontype.Boolean {
		return fmt.Errorf("undefined value should be boolean, but instead is %s", ejv.t)
	}

	if !ejv.v.(bool) {
		return fmt.Errorf("$undefined balue boolean should be true, but instead is %v", ejv.v.(bool))
	}

	return nil
}
