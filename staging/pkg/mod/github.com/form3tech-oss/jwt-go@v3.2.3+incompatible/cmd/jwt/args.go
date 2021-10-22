package main

import (
	"encoding/json"
	"fmt"
	"strings"
)

type ArgList map[string]string

func (l ArgList) String() string {
	data, _ := json.Marshal(l)
	return string(data)
}

func (l ArgList) Set(arg string) error {
	parts := strings.SplitN(arg, "=", 2)
	if len(parts) != 2 {
		return fmt.Errorf("Invalid argument '%v'.  Must use format 'key=value'. %v", arg, parts)
	}
	l[parts[0]] = parts[1]
	return nil
}
