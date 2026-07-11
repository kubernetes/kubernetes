// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package loading

type loadingError string

const (
	// ErrLoader is an error raised by the file loader utility
	ErrLoader loadingError = "loader error"
)

func (e loadingError) Error() string {
	return string(e)
}
