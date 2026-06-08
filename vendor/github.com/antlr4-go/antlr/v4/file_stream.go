// Copyright (c) 2012-2022 The ANTLR Project. All rights reserved.
// Use of this file is governed by the BSD 3-clause license that
// can be found in the LICENSE.txt file in the project root.

package antlr

import (
	"bufio"
	"os"
)

//  This is an InputStream that is loaded from a file all at once
//  when you construct the object.

type FileStream struct {
	InputStream
	filename string
}

//goland:noinspection GoUnusedExportedFunction
func NewFileStream(fileName string) (*FileStream, error) {

	f, err := os.Open(fileName)
	if err != nil {
		return nil, err
	}

	defer func(f *os.File) {
		errF := f.Close()
		if errF != nil {
		}
	}(f)

	reader := bufio.NewReader(f)
	fInfo, err := f.Stat()
	if err != nil {
		return nil, err
	}

	fs := &FileStream{
		InputStream: InputStream{
			index: 0,
			name:  fileName,
		},
		filename: fileName,
	}

	// Pre-build the buffer and read runes efficiently
	//
	fs.data = make([]rune, 0, fInfo.Size())
	for {
		r, _, err := reader.ReadRune()
		if err != nil {
			break
		}
		fs.data = append(fs.data, r)
	}
	fs.size = len(fs.data) // Size in runes

	// All done.
	//
	return fs, nil
}

func (f *FileStream) GetSourceName() string {
	return f.filename
}
