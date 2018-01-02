package jsoniter

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

func Test_bind_api_demo(t *testing.T) {
	should := require.New(t)
	val := []int{}
	err := UnmarshalFromString(`[0,1,2,3]  `, &val)
	should.Nil(err)
	should.Equal([]int{0, 1, 2, 3}, val)
}

func Test_iterator_api_demo(t *testing.T) {
	should := require.New(t)
	iter := ParseString(ConfigDefault, `[0,1,2,3]`)
	total := 0
	for iter.ReadArray() {
		total += iter.ReadInt()
	}
	should.Equal(6, total)
}

type People struct {
	Name    string
	Gender  string
	Age     int
	Address string
	Mobile  string
	Country string
	Height  int
}

func jsoniterMarshal(p *People) error {
	_, err := Marshal(p)
	if nil != err {
		return err
	}
	return nil
}
func stdMarshal(p *People) error {
	_, err := json.Marshal(p)
	if nil != err {
		return err
	}
	return nil
}

func BenchmarkJosniterMarshal(b *testing.B) {
	var p People
	p.Address = "上海市徐汇区漕宝路"
	p.Age = 30
	p.Country = "中国"
	p.Gender = "male"
	p.Height = 170
	p.Mobile = "18502120533"
	p.Name = "Elvin"
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		err := jsoniterMarshal(&p)
		if nil != err {
			b.Error(err)
		}
	}
}

func BenchmarkStdMarshal(b *testing.B) {
	var p People
	p.Address = "上海市徐汇区漕宝路"
	p.Age = 30
	p.Country = "中国"
	p.Gender = "male"
	p.Height = 170
	p.Mobile = "18502120533"
	p.Name = "Elvin"
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		err := stdMarshal(&p)
		if nil != err {
			b.Error(err)
		}
	}
}
