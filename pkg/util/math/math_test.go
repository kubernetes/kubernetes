/*
Copyright 2021 The Kubernetes Authors.

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

package math

import (
	"testing"
)

func TestMaxInt(t *testing.T) {
	type args struct {
		a int
		b int
	}
	tests := []struct {
		name string
		args args
		want int
	}{
		{"a should be bigger", args{2, 1}, 2},
		{"b should be bigger", args{1, 2}, 2},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := MaxInt(tt.args.a, tt.args.b); got != tt.want {
				t.Errorf("MaxInt() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMinInt(t *testing.T) {
	type args struct {
		a int
		b int
	}
	tests := []struct {
		name string
		args args
		want int
	}{
		{"a should be smaller", args{1, 2}, 1},
		{"b should be smaller", args{2, 1}, 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := MinInt(tt.args.a, tt.args.b); got != tt.want {
				t.Errorf("MinInt() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestBoundedInt(t *testing.T) {
	type args struct {
		value int
		lower int
		upper int
	}
	tests := []struct {
		name string
		args args
		want int
	}{
		{"unchanged", args{2, 1, 3}, 2},
		{"changed to lower", args{0, 1, 3}, 1},
		{"changed to upper", args{4, 1, 3}, 3},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := BoundedInt(tt.args.value, tt.args.lower, tt.args.upper); got != tt.want {
				t.Errorf("BoundedInt() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMaxUint8(t *testing.T) {
	type args struct {
		a uint8
		b uint8
	}
	tests := []struct {
		name string
		args args
		want uint8
	}{
		{"a should be bigger", args{2, 1}, 2},
		{"b should be bigger", args{1, 2}, 2},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := MaxUint8(tt.args.a, tt.args.b); got != tt.want {
				t.Errorf("MaxUint8() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMinUint8(t *testing.T) {
	type args struct {
		a uint8
		b uint8
	}
	tests := []struct {
		name string
		args args
		want uint8
	}{
		{"a should be smaller", args{1, 2}, 1},
		{"b should be smaller", args{2, 1}, 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := MinUint8(tt.args.a, tt.args.b); got != tt.want {
				t.Errorf("MinUint8() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestBoundedUint8(t *testing.T) {
	type args struct {
		value uint8
		lower uint8
		upper uint8
	}
	tests := []struct {
		name string
		args args
		want uint8
	}{
		{"unchanged", args{2, 1, 3}, 2},
		{"changed to lower", args{0, 1, 3}, 1},
		{"changed to upper", args{4, 1, 3}, 3},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := BoundedUint8(tt.args.value, tt.args.lower, tt.args.upper); got != tt.want {
				t.Errorf("BoundedUint8() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMaxInt32(t *testing.T) {
	type args struct {
		a int32
		b int32
	}
	tests := []struct {
		name string
		args args
		want int32
	}{
		{"a should be bigger", args{2, 1}, 2},
		{"b should be bigger", args{1, 2}, 2},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := MaxInt32(tt.args.a, tt.args.b); got != tt.want {
				t.Errorf("MaxInt32() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMinInt32(t *testing.T) {
	type args struct {
		a int32
		b int32
	}
	tests := []struct {
		name string
		args args
		want int32
	}{
		{"a should be smaller", args{1, 2}, 1},
		{"b should be smaller", args{2, 1}, 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := MinInt32(tt.args.a, tt.args.b); got != tt.want {
				t.Errorf("MinInt32() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestBoundedInt32(t *testing.T) {
	type args struct {
		value int32
		lower int32
		upper int32
	}
	tests := []struct {
		name string
		args args
		want int32
	}{
		{"unchanged", args{2, 1, 3}, 2},
		{"changed to lower", args{0, 1, 3}, 1},
		{"changed to upper", args{4, 1, 3}, 3},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := BoundedInt32(tt.args.value, tt.args.lower, tt.args.upper); got != tt.want {
				t.Errorf("BoundedInt32() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMaxInt64(t *testing.T) {
	type args struct {
		a int64
		b int64
	}
	tests := []struct {
		name string
		args args
		want int64
	}{
		{"a should be bigger", args{2, 1}, 2},
		{"b should be bigger", args{1, 2}, 2},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := MaxInt64(tt.args.a, tt.args.b); got != tt.want {
				t.Errorf("MaxInt64() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMinInt64(t *testing.T) {
	type args struct {
		a int64
		b int64
	}
	tests := []struct {
		name string
		args args
		want int64
	}{
		{"a should be smaller", args{1, 2}, 1},
		{"b should be smaller", args{2, 1}, 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := MinInt64(tt.args.a, tt.args.b); got != tt.want {
				t.Errorf("MinInt64() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestBoundedInt64(t *testing.T) {
	type args struct {
		value int64
		lower int64
		upper int64
	}
	tests := []struct {
		name string
		args args
		want int64
	}{
		{"unchanged", args{2, 1, 3}, 2},
		{"changed to lower", args{0, 1, 3}, 1},
		{"changed to upper", args{4, 1, 3}, 3},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := BoundedInt64(tt.args.value, tt.args.lower, tt.args.upper); got != tt.want {
				t.Errorf("BoundedInt64() = %v, want %v", got, tt.want)
			}
		})
	}
}
