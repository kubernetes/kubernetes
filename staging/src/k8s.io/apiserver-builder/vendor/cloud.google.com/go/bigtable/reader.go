/*
Copyright 2016 Google Inc. All Rights Reserved.

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

package bigtable

import (
	"bytes"
	"fmt"

	btpb "google.golang.org/genproto/googleapis/bigtable/v2"
)

// A Row is returned by ReadRows. The map is keyed by column family (the prefix
// of the column name before the colon). The values are the returned ReadItems
// for that column family in the order returned by Read.
type Row map[string][]ReadItem

// Key returns the row's key, or "" if the row is empty.
func (r Row) Key() string {
	for _, items := range r {
		if len(items) > 0 {
			return items[0].Row
		}
	}
	return ""
}

// A ReadItem is returned by Read. A ReadItem contains data from a specific row and column.
type ReadItem struct {
	Row, Column string
	Timestamp   Timestamp
	Value       []byte
}

// The current state of the read rows state machine.
type rrState int64

const (
	newRow rrState = iota
	rowInProgress
	cellInProgress
)

// chunkReader handles cell chunks from the read rows response and combines
// them into full Rows.
type chunkReader struct {
	state   rrState
	curKey  []byte
	curFam  string
	curQual []byte
	curTS   int64
	curVal  []byte
	curRow  Row
	lastKey string
}

// newChunkReader returns a new chunkReader for handling read rows responses.
func newChunkReader() *chunkReader {
	return &chunkReader{state: newRow}
}

// Process takes a cell chunk and returns a new Row if the given chunk
// completes a Row, or nil otherwise.
func (cr *chunkReader) Process(cc *btpb.ReadRowsResponse_CellChunk) (Row, error) {
	var row Row
	switch cr.state {
	case newRow:
		if err := cr.validateNewRow(cc); err != nil {
			return nil, err
		}

		cr.curRow = make(Row)
		cr.curKey = cc.RowKey
		cr.curFam = cc.FamilyName.Value
		cr.curQual = cc.Qualifier.Value
		cr.curTS = cc.TimestampMicros
		row = cr.handleCellValue(cc)

	case rowInProgress:
		if err := cr.validateRowInProgress(cc); err != nil {
			return nil, err
		}

		if cc.GetResetRow() {
			cr.resetToNewRow()
			return nil, nil
		}

		if cc.FamilyName != nil {
			cr.curFam = cc.FamilyName.Value
		}
		if cc.Qualifier != nil {
			cr.curQual = cc.Qualifier.Value
		}
		cr.curTS = cc.TimestampMicros
		row = cr.handleCellValue(cc)

	case cellInProgress:
		if err := cr.validateCellInProgress(cc); err != nil {
			return nil, err
		}
		if cc.GetResetRow() {
			cr.resetToNewRow()
			return nil, nil
		}
		row = cr.handleCellValue(cc)
	}

	return row, nil
}

// Close must be called after all cell chunks from the response
// have been processed. An error will be returned if the reader is
// in an invalid state, in which case the error should be propagated to the caller.
func (cr *chunkReader) Close() error {
	if cr.state != newRow {
		return fmt.Errorf("invalid state for end of stream %q", cr.state)
	}
	return nil
}

// handleCellValue returns a Row if the cell value includes a commit, otherwise nil.
func (cr *chunkReader) handleCellValue(cc *btpb.ReadRowsResponse_CellChunk) Row {
	if cc.ValueSize > 0 {
		// ValueSize is specified so expect a split value of ValueSize bytes
		if cr.curVal == nil {
			cr.curVal = make([]byte, 0, cc.ValueSize)
		}
		cr.curVal = append(cr.curVal, cc.Value...)
		cr.state = cellInProgress
	} else {
		// This cell is either the complete value or the last chunk of a split
		if cr.curVal == nil {
			cr.curVal = cc.Value
		} else {
			cr.curVal = append(cr.curVal, cc.Value...)
		}
		cr.finishCell()

		if cc.GetCommitRow() {
			return cr.commitRow()
		} else {
			cr.state = rowInProgress
		}
	}

	return nil
}

func (cr *chunkReader) finishCell() {
	ri := ReadItem{
		Row:       string(cr.curKey),
		Column:    fmt.Sprintf("%s:%s", cr.curFam, cr.curQual),
		Timestamp: Timestamp(cr.curTS),
		Value:     cr.curVal,
	}
	cr.curRow[cr.curFam] = append(cr.curRow[cr.curFam], ri)
	cr.curVal = nil
}

func (cr *chunkReader) commitRow() Row {
	row := cr.curRow
	cr.lastKey = cr.curRow.Key()
	cr.resetToNewRow()
	return row
}

func (cr *chunkReader) resetToNewRow() {
	cr.curKey = nil
	cr.curFam = ""
	cr.curQual = nil
	cr.curVal = nil
	cr.curRow = nil
	cr.curTS = 0
	cr.state = newRow
}

func (cr *chunkReader) validateNewRow(cc *btpb.ReadRowsResponse_CellChunk) error {
	if cc.GetResetRow() {
		return fmt.Errorf("reset_row not allowed between rows")
	}
	if cc.RowKey == nil || cc.FamilyName == nil || cc.Qualifier == nil {
		return fmt.Errorf("missing key field for new row %v", cc)
	}
	if cr.lastKey != "" && cr.lastKey >= string(cc.RowKey) {
		return fmt.Errorf("out of order row key: %q, %q", cr.lastKey, string(cc.RowKey))
	}
	return nil
}

func (cr *chunkReader) validateRowInProgress(cc *btpb.ReadRowsResponse_CellChunk) error {
	if err := cr.validateRowStatus(cc); err != nil {
		return err
	}
	if cc.RowKey != nil && !bytes.Equal(cc.RowKey, cr.curKey) {
		return fmt.Errorf("received new row key %q during existing row %q", cc.RowKey, cr.curKey)
	}
	if cc.FamilyName != nil && cc.Qualifier == nil {
		return fmt.Errorf("family name %q specified without a qualifier", cc.FamilyName)
	}
	return nil
}

func (cr *chunkReader) validateCellInProgress(cc *btpb.ReadRowsResponse_CellChunk) error {
	if err := cr.validateRowStatus(cc); err != nil {
		return err
	}
	if cr.curVal == nil {
		return fmt.Errorf("no cached cell while CELL_IN_PROGRESS %v", cc)
	}
	if cc.GetResetRow() == false && cr.isAnyKeyPresent(cc) {
		return fmt.Errorf("cell key components found while CELL_IN_PROGRESS %v", cc)
	}
	return nil
}

func (cr *chunkReader) isAnyKeyPresent(cc *btpb.ReadRowsResponse_CellChunk) bool {
	return cc.RowKey != nil ||
		cc.FamilyName != nil ||
		cc.Qualifier != nil ||
		cc.TimestampMicros != 0
}

// Validate a RowStatus, commit or reset, if present.
func (cr *chunkReader) validateRowStatus(cc *btpb.ReadRowsResponse_CellChunk) error {
	// Resets can't be specified with any other part of a cell
	if cc.GetResetRow() && (cr.isAnyKeyPresent(cc) ||
		cc.Value != nil ||
		cc.ValueSize != 0 ||
		cc.Labels != nil) {
		return fmt.Errorf("reset must not be specified with other fields %v", cc)
	}
	if cc.GetCommitRow() && cc.ValueSize > 0 {
		return fmt.Errorf("commit row found in between chunks in a cell")
	}
	return nil
}
