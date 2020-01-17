// Copyright 2015 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package runtime

import (
	"bytes"
	"encoding/csv"
	"errors"
	"io"
)

// CSVConsumer creates a new CSV consumer
func CSVConsumer() Consumer {
	return ConsumerFunc(func(reader io.Reader, data interface{}) error {
		if reader == nil {
			return errors.New("CSVConsumer requires a reader")
		}

		csvReader := csv.NewReader(reader)
		writer, ok := data.(io.Writer)
		if !ok {
			return errors.New("data type must be io.Writer")
		}
		csvWriter := csv.NewWriter(writer)
		records, err := csvReader.ReadAll()
		if err != nil {
			return err
		}
		for _, r := range records {
			if err := csvWriter.Write(r); err != nil {
				return err
			}
		}
		csvWriter.Flush()
		return nil
	})
}

// CSVProducer creates a new CSV producer
func CSVProducer() Producer {
	return ProducerFunc(func(writer io.Writer, data interface{}) error {
		if writer == nil {
			return errors.New("CSVProducer requires a writer")
		}

		dataBytes, ok := data.([]byte)
		if !ok {
			return errors.New("data type must be byte array")
		}

		csvReader := csv.NewReader(bytes.NewBuffer(dataBytes))
		records, err := csvReader.ReadAll()
		if err != nil {
			return err
		}
		csvWriter := csv.NewWriter(writer)
		for _, r := range records {
			if err := csvWriter.Write(r); err != nil {
				return err
			}
		}
		csvWriter.Flush()
		return nil
	})
}
