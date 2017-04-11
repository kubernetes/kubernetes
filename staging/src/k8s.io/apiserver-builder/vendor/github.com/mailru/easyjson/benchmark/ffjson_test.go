// +build use_ffjson

package benchmark

import (
	"testing"

	"github.com/pquerna/ffjson/ffjson"
)

func BenchmarkFF_Unmarshal_M(b *testing.B) {
	b.SetBytes(int64(len(largeStructText)))
	for i := 0; i < b.N; i++ {
		var s LargeStruct
		err := ffjson.UnmarshalFast(largeStructText, &s)
		if err != nil {
			b.Error(err)
		}
	}
}

func BenchmarkFF_Unmarshal_S(b *testing.B) {
	for i := 0; i < b.N; i++ {
		var s Entities
		err := ffjson.UnmarshalFast(smallStructText, &s)
		if err != nil {
			b.Error(err)
		}
	}
	b.SetBytes(int64(len(smallStructText)))
}

func BenchmarkFF_Marshal_M(b *testing.B) {
	var l int64
	for i := 0; i < b.N; i++ {
		data, err := ffjson.MarshalFast(&largeStructData)
		if err != nil {
			b.Error(err)
		}
		l = int64(len(data))
	}
	b.SetBytes(l)
}

func BenchmarkFF_Marshal_S(b *testing.B) {
	var l int64
	for i := 0; i < b.N; i++ {
		data, err := ffjson.MarshalFast(&smallStructData)
		if err != nil {
			b.Error(err)
		}
		l = int64(len(data))
	}
	b.SetBytes(l)
}

func BenchmarkFF_Marshal_M_Pool(b *testing.B) {
	var l int64
	for i := 0; i < b.N; i++ {
		data, err := ffjson.MarshalFast(&largeStructData)
		if err != nil {
			b.Error(err)
		}
		l = int64(len(data))
		ffjson.Pool(data)
	}
	b.SetBytes(l)
}

func BenchmarkFF_Marshal_L(b *testing.B) {
	var l int64
	for i := 0; i < b.N; i++ {
		data, err := ffjson.MarshalFast(&xlStructData)
		if err != nil {
			b.Error(err)
		}
		l = int64(len(data))
	}
	b.SetBytes(l)
}

func BenchmarkFF_Marshal_L_Pool(b *testing.B) {
	var l int64
	for i := 0; i < b.N; i++ {
		data, err := ffjson.MarshalFast(&xlStructData)
		if err != nil {
			b.Error(err)
		}
		l = int64(len(data))
		ffjson.Pool(data)
	}
	b.SetBytes(l)
}

func BenchmarkFF_Marshal_L_Pool_Parallel(b *testing.B) {
	var l int64
	for i := 0; i < b.N; i++ {
		data, err := ffjson.MarshalFast(&xlStructData)
		if err != nil {
			b.Error(err)
		}
		l = int64(len(data))
		ffjson.Pool(data)
	}
	b.SetBytes(l)
}
func BenchmarkFF_Marshal_M_Pool_Parallel(b *testing.B) {
	var l int64
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			data, err := ffjson.MarshalFast(&largeStructData)
			if err != nil {
				b.Error(err)
			}
			l = int64(len(data))
			ffjson.Pool(data)
		}
	})
	b.SetBytes(l)
}

func BenchmarkFF_Marshal_S_Pool(b *testing.B) {
	var l int64
	for i := 0; i < b.N; i++ {
		data, err := ffjson.MarshalFast(&smallStructData)
		if err != nil {
			b.Error(err)
		}
		l = int64(len(data))
		ffjson.Pool(data)
	}
	b.SetBytes(l)
}

func BenchmarkFF_Marshal_S_Pool_Parallel(b *testing.B) {
	var l int64
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			data, err := ffjson.MarshalFast(&smallStructData)
			if err != nil {
				b.Error(err)
			}
			l = int64(len(data))
			ffjson.Pool(data)
		}
	})
	b.SetBytes(l)
}

func BenchmarkFF_Marshal_S_Parallel(b *testing.B) {
	var l int64
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			data, err := ffjson.MarshalFast(&smallStructData)
			if err != nil {
				b.Error(err)
			}
			l = int64(len(data))
		}
	})
	b.SetBytes(l)
}

func BenchmarkFF_Marshal_M_Parallel(b *testing.B) {
	var l int64
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			data, err := ffjson.MarshalFast(&largeStructData)
			if err != nil {
				b.Error(err)
			}
			l = int64(len(data))
		}
	})
	b.SetBytes(l)
}

func BenchmarkFF_Marshal_L_Parallel(b *testing.B) {
	var l int64
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			data, err := ffjson.MarshalFast(&xlStructData)
			if err != nil {
				b.Error(err)
			}
			l = int64(len(data))
		}
	})
	b.SetBytes(l)
}
