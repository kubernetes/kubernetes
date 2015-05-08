package types

import (
	"encoding/json"
	"errors"
	"fmt"
)

type EventHandler struct {
	Name string `json:"name"`
	Exec Exec   `json:"exec"`
}

type eventHandler EventHandler

func (e EventHandler) assertValid() error {
	s := e.Name
	switch s {
	case "pre-start", "post-stop":
		return nil
	case "":
		return errors.New(`eventHandler "name" cannot be empty`)
	default:
		return fmt.Errorf(`bad eventHandler "name": %q`, s)
	}
}

func (e EventHandler) MarshalJSON() ([]byte, error) {
	if err := e.assertValid(); err != nil {
		return nil, err
	}
	return json.Marshal(eventHandler(e))
}

func (e *EventHandler) UnmarshalJSON(data []byte) error {
	var je eventHandler
	err := json.Unmarshal(data, &je)
	if err != nil {
		return err
	}
	ne := EventHandler(je)
	if err := ne.assertValid(); err != nil {
		return err
	}
	*e = ne
	return nil
}
