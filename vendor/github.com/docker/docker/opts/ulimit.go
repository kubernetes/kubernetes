package opts

import (
	"fmt"

	"github.com/docker/docker/pkg/ulimit"
)

type UlimitOpt struct {
	values map[string]*ulimit.Ulimit
}

func NewUlimitOpt(ref map[string]*ulimit.Ulimit) *UlimitOpt {
	return &UlimitOpt{ref}
}

func (o *UlimitOpt) Set(val string) error {
	l, err := ulimit.Parse(val)
	if err != nil {
		return err
	}

	o.values[l.Name] = l

	return nil
}

func (o *UlimitOpt) String() string {
	var out []string
	for _, v := range o.values {
		out = append(out, v.String())
	}

	return fmt.Sprintf("%v", out)
}

func (o *UlimitOpt) GetList() []*ulimit.Ulimit {
	var ulimits []*ulimit.Ulimit
	for _, v := range o.values {
		ulimits = append(ulimits, v)
	}

	return ulimits
}
