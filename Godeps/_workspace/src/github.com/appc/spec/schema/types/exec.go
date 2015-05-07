package types

import (
	"encoding/json"
	"errors"
	"path/filepath"
)

type Exec []string

type exec Exec

func (e Exec) assertValid() error {
	if len(e) < 1 {
		return errors.New(`Exec cannot be empty`)
	}
	if !filepath.IsAbs(e[0]) {
		return errors.New(`Exec[0] must be absolute path`)
	}
	return nil
}

func (e Exec) MarshalJSON() ([]byte, error) {
	if err := e.assertValid(); err != nil {
		return nil, err
	}
	return json.Marshal(exec(e))
}

func (e *Exec) UnmarshalJSON(data []byte) error {
	var je exec
	err := json.Unmarshal(data, &je)
	if err != nil {
		return err
	}
	ne := Exec(je)
	if err := ne.assertValid(); err != nil {
		return err
	}
	*e = ne
	return nil
}
