package cgroups

import (
	"fmt"
)

type IfPrioMap struct {
	Interface string `json:"interface"`
	Priority  int64  `json:"priority"`
}

func (i *IfPrioMap) CgroupString() string {
	return fmt.Sprintf("%s %d", i.Interface, i.Priority)
}
